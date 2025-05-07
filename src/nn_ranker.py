import json
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import re

# --- Handle Input Data ---
def load_abstracts(cache_file: str):
    """
    Load abstracts from JSON file
    Return: Dict[pubmed_id, abstract] 
    """
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load abstracts: {e}")
        return {}
    
def load_query_data(json_file: str):
    """
    Load training data from the JSON training file
    Return: List[Dict[questions]]
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)['questions']
    except Exception as e:
        print(f"Failed to load abstracts: {e}")
        return {}
    
# --- Model ---
def compute_doc_embeddings(abstracts_dict, model, batch_size):
    doc_ids = list(abstracts_dict.keys())
    texts = [abstracts_dict[doc_id] for doc_id in doc_ids]
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True)
    return dict(zip(doc_ids, embeddings))

class PubMedQueryDataset(Dataset):
    def __init__(self, training_data):
        self.examples = training_data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "id": item["id"],
            "query": item["body"],
            "doc_ids": [url.split("/")[-1] for url in item.get("documents", [])]
        }

def collate_fn(batch):
    """Custom processing for PyTorch Dataset"""
    queries = [item["query"] for item in batch]
    ids = [item["id"] for item in batch]
    doc_ids = [item["doc_ids"] for item in batch]
    return ids, queries, doc_ids

def extract_snippets(model, query_embedding, abstract, doc_id, threshold):
    """Extract relevant sentences with character offsets """
    # Create sentence embeddings
    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())
    if not sentences:
        return []
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Rank sentences by similarity, extract top n
    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]  # shape: (num_sentences,)
    
    snippet_candidates = []
    for idx, score_tensor in enumerate(similarities):
        if score_tensor.item() < threshold: # keep only if similar enough -> arg
            continue

        snippet_text = sentences[idx]
        char_start = abstract.find(snippet_text)
        char_end = char_start + len(snippet_text)

        snippet_candidates.append({
            "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}",
            "beginSection": "abstract",
            "endSection": "abstract",
            "offsetInBeginSection": char_start,
            "offsetInEndSection": char_end,
            "text": snippet_text,
            "score": score_tensor.item()
        })


    return snippet_candidates

def compute_metrics(retrieved_ids, ground_truth):
    """Compute evaluation metrics for retrieved documents"""
    k = len(retrieved_ids)
    hits = sum(1 for doc_id in retrieved_ids if doc_id in ground_truth)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute MAP
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in ground_truth:
            num_hits += 1
            score += num_hits / (i + 1)
    average_precision = score / len(ground_truth) if ground_truth else 0.0

    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'average_precision': average_precision
    }

def run_retrieval(model_name, query_data, abstracts_dict, debug=False, batch_size=8):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name)
    model.to(device)
    print(f"Model loaded on device: {device}")

    # Precompute document embeddings
    print("Computing abstract embeddings...")
    doc_embedding_map = compute_doc_embeddings(abstracts_dict, model, batch_size)
    doc_ids_all = list(doc_embedding_map.keys())
    doc_embeddings_all = torch.stack([doc_embedding_map[doc_id] for doc_id in doc_ids_all]).to(device) # shape: (num_docs, embedding_dim)

    # Dataset and loader
    dataset = PubMedQueryDataset(query_data)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Initialize output
    output = {"questions": []}

    # If ground truth available, initialize metrics
    compute_eval = any(item.get("documents", None) for item in query_data)
    total_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "average_precision": 0.0
    } if compute_eval else None
    num_queries = 0
    
    for batch_idx, (ids, queries, ground_truth_ids) in tqdm(enumerate(loader), total=len(loader)):
        # If in debug mode, run only two batches
        if debug and batch_idx >= 2:
            break
        query_embeddings = model.encode(queries, convert_to_tensor=True).to(device) # shape: (batch_size, embedding_dim)
        cos_scores = util.cos_sim(query_embeddings, doc_embeddings_all)  # shape: (batch_size, num_docs)

        for i, qid in enumerate(ids):
            scores = cos_scores[i]
            top_k_idx = torch.topk(scores, k=10).indices
            ranked_doc_ids = [doc_ids_all[j] for j in top_k_idx]

            if compute_eval:
                metrics = compute_metrics(ranked_doc_ids, ground_truth_ids[i])
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_queries += 1

            # Extract snippets
            snippet_candidates = []
            for doc_id in ranked_doc_ids:
                abstract = abstracts_dict.get(doc_id, "")
                if not abstract.strip():
                    continue
                snippets = extract_snippets(model, query_embeddings[i], abstract, doc_id, threshold=0.6)
                snippet_candidates.extend(snippets)
            
            # Keep only top 10
            top_snippets = sorted(snippet_candidates, key=lambda x: x["score"], reverse=True)[:10]
            for snippet in top_snippets:
                snippet.pop("score", None)

            output["questions"].append({
                "id": qid,
                "documents": [f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}" for doc_id in ranked_doc_ids],
                "snippets": snippets
            })

    if compute_eval:
        avg_metrics = {key: val / num_queries for key, val in total_metrics.items()}
    else:
        avg_metrics = {}

    return output, avg_metrics

def save_results(model_name, results, metrics):
    # Save metrics to txt if available
    if metrics:
        with open(f"../results/{model_name}_eval.txt", "w", encoding="utf-8") as f:
            f.write("Evaluation Results:\n")
            f.write(f"MAP:       {metrics['average_precision']:.4f}\n")
            f.write(f"P@10:      {metrics['precision']:.4f}\n")
            f.write(f"R@10:      {metrics['recall']:.4f}\n")
            f.write(f"F1@10:     {metrics['f1']:.4f}\n")
        print(f"Evaluation metrics saved to ../results/{model_name}_eval.txt")
    else:
        print("Evaluation metrics not computed.")

    # Save results
    with open(f"../results/{model_name}_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Output saved to ../results/{model_name}_output.json")


# ------
if __name__ == "__main__":

    debug = False

    print("Loading data...")
    abstracts_dict = load_abstracts("../data/pubmed_abstracts.json")
    #training_data = load_query_data("../data/training13b.json")
    test_data = load_query_data("../data/BioASQ-task13bPhaseA-testset4")

    # run ranking
    #model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    #model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    model_name = 'pritamdeka/S-PubMedBert-MS-MARCO'

    results, metrics = run_retrieval(
        model_name=model_name,
        query_data=test_data,
        abstracts_dict=abstracts_dict,
        debug=debug,
        batch_size=8
    )

    # Save results to json ang aggregated metrics to txt
    save_results("nn_pubmed_test4", results, metrics)


    