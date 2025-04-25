import torch
import os
import orjson
import transformers
from torch.utils.data import Dataset, DataLoader
from torch.nn import MarginRankingLoss
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import json
from typing import List, Tuple
import random
from collections import defaultdict
import time
import warnings
from functools import wraps


# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.to(device)
print(f"Training on device: {device}")

# Configs
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 256
LEARNING_RATE = 2e-5
TOP_K = 10
transformers.logging.set_verbosity_error()


# --- Data Creation from BioASQ-style files ---


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds.")
        return result
    return wrapper


def extract_pubmed_id(url: str) -> str:
    return url.strip().split("/")[-1]


def build_gold_labels(training_path: str) -> dict:
    with open(training_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    qrels = {}
    for q in train_data['questions']:
        qrels[q['body']] = [extract_pubmed_id(url) for url in q.get('documents', [])]
    return qrels


def build_bm25_data(result_path: str) -> dict:
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    bm25_dict = defaultdict(lambda: {"query": "", "documents": defaultdict(str)})

    for q in result_data['questions']:
        qid = q['id']
        for snippet in q.get('snippets', []):
            doc_id = extract_pubmed_id(snippet['document'])
            bm25_dict[qid]['documents'][doc_id] += " " + snippet['text']

    return bm25_dict


@timer
def create_train_triples_from_bioasq(training_path: str, result_path: str, output_path: str):
    qrels = build_gold_labels(training_path)
    bm25_data = build_bm25_data(result_path)

    with open(training_path, 'r', encoding='utf-8') as f:
        questions = {q['id']: q['body'] for q in json.load(f)['questions']}

    triples = []
    for qid, qtext in questions.items():
        if qid not in bm25_data or not bm25_data[qid]['documents']:
            continue

        relevant = set(qrels.get(qtext, []))
        candidates = bm25_data[qid]['documents']

        positives = [doc for doc in candidates if doc in relevant]
        negatives = [doc for doc in candidates if doc not in relevant]

        for pos_id in positives:
            if not negatives:
                continue
            neg_id = random.choice(negatives)
            triples.append({
                "query": qtext,
                "positive": candidates[pos_id],
                "negative": candidates[neg_id]
            })

    with open(output_path, 'w') as f:
        json.dump(triples, f, indent=2)


@timer
def create_inference_input_from_result(result_path: str, training_path: str, output_path: str):
    bm25_data = build_bm25_data(result_path)
    with open(training_path, 'r', encoding='utf-8') as f:
        questions = {q['id']: q['body'] for q in json.load(f)['questions']}

    pairs = []
    for qid, qtext in questions.items():
        for doc_id, snippet in bm25_data[qid]['documents'].items():
            pairs.append({"query": qtext, "document": snippet, "doc_id": doc_id})

    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)


# Load training data
@timer
def load_training_data(path: str, test_run: bool = False) -> List[Tuple[str, str, str]]:
    with open(path, 'r') as f:
        data = json.load(f)
    if test_run:
        warnings.warn("Running smoke test! Only 100 tuples will be used for training.")
        data = data[:100]
    return [(x['query'], x['positive'], x['negative']) for x in data]


@timer
def load_inference_data(path: str) -> List[Tuple[str, str]]:
    with open(path, 'r') as f:
        data = json.load(f)
    return [(x['query'], x['document']) for x in data]


class TripletDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        q, pos, neg = self.triples[idx]
        pos_enc = tokenizer(q, pos, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        neg_enc = tokenizer(q, neg, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        return pos_enc, neg_enc


class InferenceDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, d = self.pairs[idx]['query'], self.pairs[idx]['document']
        enc = self.tokenizer(q, d, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

        return {
            'query': q,
            'doc_id': self.pairs[idx]['doc_id'],
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }


def train_re_ranking(test_run: bool = False):
    loss_fn = MarginRankingLoss(margin=1.0)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    train_data = load_training_data(os.path.join('data', 'train_triples.json'), test_run=test_run)
    dataset = TripletDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        for step, (pos_enc, neg_enc) in enumerate(dataloader):
            input_ids_pos = pos_enc['input_ids'].squeeze(1).to(device)
            attention_mask_pos = pos_enc['attention_mask'].squeeze(1).to(device)

            input_ids_neg = neg_enc['input_ids'].squeeze(1).to(device)
            attention_mask_neg = neg_enc['attention_mask'].squeeze(1).to(device)

            outputs_pos = model(input_ids=input_ids_pos, attention_mask=attention_mask_pos).logits.squeeze()
            outputs_neg = model(input_ids=input_ids_neg, attention_mask=attention_mask_neg).logits.squeeze()

            targets = torch.ones(outputs_pos.size()).to(device)
            loss = loss_fn(outputs_pos, outputs_neg, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        print(f"Epoch {epoch+1} took {epoch_duration:.4f} seconds.")
    model.save_pretrained('bert_reranker_model')
    tokenizer.save_pretrained('bert_reranker_model')


@timer
def run_inference(model_path: str,
                  inference_data_path: str,
                  output_path: str,
                  batch_size: int = 16,
                  num_workers: int = 4):
    print("Loading model for inference...")
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()

    with open(inference_data_path, 'rb') as f:
        raw_data = orjson.loads(f.read())

    dataset = InferenceDataset(raw_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    results = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu().numpy()

            for i in range(len(batch['query'])):
                results.append({
                    "query": batch['query'][i],
                    "doc_id": batch['doc_id'][i],
                    "score": output[i].tolist()
                })

    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(results))


@timer
def evaluate_reranker(predictions_path: str, training_path: str):
    with open(predictions_path, 'r') as f:
        preds = json.load(f)

    qrels = build_gold_labels(training_path)
    rankings = defaultdict(list)
    for p in preds:
        query_key = " ".join(p['query']) if isinstance(p['query'], list) else p['query']
        rankings[query_key].append((p['doc_id'], p['score']))

    def compute_metrics(ranked, relevant):
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        retrieved = [doc[0][0] for doc in ranked[:TOP_K]]
        rel_set = set(relevant)
        hits = [1 if doc in rel_set else 0 for doc in retrieved]

        p10 = sum(hits) / TOP_K
        r10 = sum(hits) / len(rel_set) if rel_set else 0
        f1 = (2 * p10 * r10 / (p10 + r10)) if (p10 + r10) else 0

        ap = 0.0
        num_hits = 0
        for i, doc in enumerate(retrieved):
            if doc in rel_set:
                num_hits += 1
                ap += num_hits / (i + 1)

        ap = ap / len(rel_set) if rel_set else 0

        return ap, p10, r10, f1

    total_ap, total_p, total_r, total_f1 = 0, 0, 0, 0
    count = 0
    for q in rankings:
        if q not in qrels:
            continue
        ap, p, r, f1 = compute_metrics(rankings[q], qrels[q])
        total_ap += ap
        total_p += p
        total_r += r
        total_f1 += f1
        count += 1

    print("Evaluation Results:")
    print(f"MAP:   {total_ap / count:.4f}")
    print(f"P@10:  {total_p / count:.4f}")
    print(f"R@10:  {total_r / count:.4f}")
    print(f"F1@10: {total_f1 / count:.4f}")


# Example usage
if __name__ == "__main__":
    SMOKE_TEST_RUN = True
    create_train_triples_from_bioasq(
        os.path.join('data', 'training13b.json'),
        os.path.join('results', 'bm25_predictions.json'),
        os.path.join('data', 'train_triples.json')
    )

    create_inference_input_from_result(
        os.path.join('results', 'bm25_predictions.json'),
        os.path.join('data', 'training13b.json'),
        os.path.join('data', 'inference_input.json')
    )

    train_re_ranking(test_run=SMOKE_TEST_RUN)

    run_inference(
        'bert_reranker_model',
        os.path.join('data', 'inference_input.json'),
        os.path.join('results', 'rerank_predictions.json')
    )

    evaluate_reranker(
        os.path.join('results', 'rerank_predictions.json'),
        os.path.join('data', 'training13b.json')
    )
