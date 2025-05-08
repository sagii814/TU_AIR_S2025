import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from typing import List, Tuple, Dict

# Configuration
BIOASQ_PATH = "../data/training13b.json"
ABSTRACT_CACHE = "../data/pubmed_abstracts.json"
OUTPUT_JSON = "../results/bm25_predictions.json"
TESTSET_PATH = "../data/BioASQ-task13bPhaseA-testset4"
TEST_OUTPUT_JSON = "../results/bm25_test_predictions.json"
TOP_K_DOCUMENTS = 10
TOP_N_SNIPPETS = 10

def load_cached_abstracts(cache_file: str) -> Dict[str, str]:
    """Load abstracts from JSON file"""
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Cache file not found at {cache_file}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {cache_file}")
        return {}

def build_bm25_index(abstracts_dict: Dict[str, str]) -> Tuple[BM25Okapi, List[str]]:
    """Build a BM25 index from the abstracts dictionary"""
    doc_ids = list(abstracts_dict.keys())
    corpus = [(abstracts_dict[pid] or "").lower().split() for pid in doc_ids]
    bm25 = BM25Okapi(corpus)
    return bm25, doc_ids

def load_bioasq_questions(json_file: str) -> List[dict]:
    """Load BioASQ questions from the JSON training file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)['questions']
    except FileNotFoundError:
        print(f"Error: BioASQ questions file not found at {json_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}")
        return []

def get_pubmed_id_from_url(url: str) -> str:
    """Extract the PubMed ID from a full URL"""
    return url.split("/")[-1]

def simple_sent_tokenize(text: str) -> List[str]:
    """Split text into sentences using basic punctuation rules"""
    return [s.strip() for s in re.split(r'(?<!\\w\.[a-z])(?<![A-Z][a-z]\.)\s(?=[A-Z])', text) if s.strip()]

def bm25_retrieve_top_k(bm25: BM25Okapi, doc_ids: List[str], query: str, k: int = 10) -> Tuple[List[str], List[float]]:
    """Retrieve top-k documents using BM25 given a query"""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [doc_ids[i] for i in top_indices], [scores[i] for i in top_indices]

def extract_snippets(query: str, abstract: str, doc_id: str, top_n: int = 10) -> List[Dict]:
    """Extract top-n relevant snippets based on character offsets"""
    if not abstract:
        return []

    query_tokens = set(query.lower().split())
    candidate_snippets = []

    for token in query_tokens:
        for match in re.finditer(re.escape(token), abstract.lower()):
            start = match.start()
            end = match.end()

            # Expand snippet boundaries (simple context - can be refined)
            context_before = abstract[:start].rfind('. ')
            context_after = abstract[end:].find('. ')

            begin_offset = max(0, context_before + 2) if context_before != -1 else 0
            end_offset = (end + context_after) if context_after != -1 else len(abstract)
            snippet_text = abstract[begin_offset:end_offset].strip()

            snippet_data = {
                "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}",
                "beginSection": "abstract",
                "endSection": "abstract",
                "offsetInBeginSection": begin_offset,
                "offsetInEndSection": end_offset,
                "text": snippet_text
            }
            candidate_snippets.append((snippet_data, 1.0)) # Simple score for now

    # Add title as a potential snippet
    title_match = re.search(r"<article-title>(.*?)</article-title>", abstract) # If title is marked up
    if title_match:
        title_text = title_match.group(1).strip()
        if any(qt in title_text.lower() for qt in query_tokens):
            begin_title = abstract.find(title_match.group(0)) + abstract.find(title_text)
            end_title = begin_title + len(title_text)
            candidate_snippets.append({
                "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}",
                "beginSection": "title",
                "endSection": "title",
                "offsetInBeginSection": begin_title,
                "offsetInEndSection": end_title,
                "text": title_text
            }, 0.8) # Lower score for title initially

    # Rank and take top n unique snippets (can refine scoring)
    unique_snippets = {}
    for snippet, score in candidate_snippets:
        key = f"{snippet['beginSection']}-{snippet['offsetInBeginSection']}-{snippet['offsetInEndSection']}"
        unique_snippets[key] = snippet

    sorted_snippets = list(unique_snippets.values())[:top_n]
    return sorted_snippets

def get_gold_doc_ids(q: dict) -> List[str]:
    """Extract the list of gold-standard PubMed IDs from a question"""
    return [get_pubmed_id_from_url(url) for url in q.get("documents", []) if "pubmed" in url]

def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    """Calculate precision at k"""
    retrieved_k = retrieved[:k]
    rel_set = set(relevant)
    hits = sum(1 for doc_id in retrieved_k if doc_id in rel_set)
    return hits / k if k > 0 else 0.0

def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    """Calculate recall at k"""
    retrieved_k = retrieved[:k]
    rel_set = set(relevant)
    hits = sum(1 for doc_id in retrieved_k if doc_id in rel_set)
    return hits / len(rel_set) if rel_set else 0.0

def f1_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    """Calculate F1-score at k"""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate average precision for a single query."""
    rel_set = set(relevant)
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in rel_set:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(rel_set) if rel_set else 0.0


if __name__ == "__main__":
    #For the Training set

    """
    print("Loading cached abstracts...")
    abstracts_dict = load_cached_abstracts(ABSTRACT_CACHE)
    if not abstracts_dict:
        exit()

    print("Building BM25 index...")
    bm25, doc_ids = build_bm25_index(abstracts_dict)

    print("Loading BioASQ questions...")
    questions = load_bioasq_questions(BIOASQ_PATH)
    if not questions:
        exit()

    total_p, total_r, total_f1, total_ap, count = 0, 0, 0, 0, 0
    predictions = []

    print("Running BM25 retrieval and evaluation...")
    for q in tqdm(questions):
        query = q['body']
        qid = q['id']
        gold_ids = get_gold_doc_ids(q)
        if not gold_ids:
            continue

        retrieved_ids, _ = bm25_retrieve_top_k(bm25, doc_ids, query, k=TOP_K_DOCUMENTS)

        precision = precision_at_k(retrieved_ids, gold_ids, k=TOP_K_DOCUMENTS)
        recall = recall_at_k(retrieved_ids, gold_ids, k=TOP_K_DOCUMENTS)
        f1 = f1_at_k(retrieved_ids, gold_ids, k=TOP_K_DOCUMENTS)
        ap = average_precision(retrieved_ids, gold_ids)

        total_p += precision
        total_r += recall
        total_f1 += f1
        total_ap += ap
        count += 1

        all_snippets = []
        for doc_id in retrieved_ids:
            abstract = abstracts_dict.get(doc_id, "")
            snippets = extract_snippets(query, abstract, doc_id, top_n=TOP_N_SNIPPETS)
            all_snippets.extend(snippets)

        # Take the top N snippets overall (you'll need to implement proper ranking here)
        top_snippets = all_snippets[:TOP_N_SNIPPETS]

        predictions.append({
            "id": qid,
            "documents": [f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}" for doc_id in retrieved_ids],
            "snippets": top_snippets
        })

    if count > 0:
        print("\nEvaluation Results:")
        print(f"MAP:       {total_ap / count:.4f}")
        print(f"P@{TOP_K_DOCUMENTS}:   {total_p / count:.4f}")
        print(f"R@{TOP_K_DOCUMENTS}:   {total_r / count:.4f}")
        print(f"F1@{TOP_K_DOCUMENTS}:  {total_f1 / count:.4f}")
    else:
        print("\nNo questions with gold standard documents found for evaluation.")

    print(f"\nSaving predictions to {OUTPUT_JSON}")
    Path("../results").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({"questions": predictions}, f, indent=2, ensure_ascii=False)

    print("Done")
    """

    #For the Batch 4 Test Set: BioASQ-task13bPhaseA-testset4

    print("Loading cached abstracts...")
    abstracts_dict = load_cached_abstracts(ABSTRACT_CACHE)
    if not abstracts_dict:
        exit()

    print("Building BM25 index...")
    bm25, doc_ids = build_bm25_index(abstracts_dict)

    print("Loading BioASQ test questions...")
    test_questions = load_bioasq_questions(TESTSET_PATH)

    total_p, total_r, total_f1, total_ap, count = 0, 0, 0, 0, 0
    predictions = []

    print("Running BM25 retrieval and evaluation on test questions...")
    for q in tqdm(test_questions):
        query = q['body']
        qid = q['id']
        gold_ids = get_gold_doc_ids(q)

        retrieved_ids, _ = bm25_retrieve_top_k(bm25, doc_ids, query, k=TOP_K_DOCUMENTS)

        if gold_ids:
            precision = precision_at_k(retrieved_ids, gold_ids, k=TOP_K_DOCUMENTS)
            recall = recall_at_k(retrieved_ids, gold_ids, k=TOP_K_DOCUMENTS)
            f1 = f1_at_k(retrieved_ids, gold_ids, k=TOP_K_DOCUMENTS)
            ap = average_precision(retrieved_ids, gold_ids)
            total_p += precision
            total_r += recall
            total_f1 += f1
            total_ap += ap
            count += 1

        all_snippets = []
        for doc_id in retrieved_ids:
            abstract = abstracts_dict.get(doc_id, "")
            snippets = extract_snippets(query, abstract, doc_id, top_n=TOP_N_SNIPPETS)
            all_snippets.extend(snippets)

        top_snippets = all_snippets[:TOP_N_SNIPPETS]

        predictions.append({
            "id": qid,
            "documents": [f"http://www.ncbi.nlm.nih.gov/pubmed/{doc_id}" for doc_id in retrieved_ids],
            "snippets": top_snippets
        })

    if count > 0:
        print("\nEvaluation Results on test set:")
        print(f"MAP:       {total_ap / count:.4f}")
        print(f"P@{TOP_K_DOCUMENTS}:   {total_p / count:.4f}")
        print(f"R@{TOP_K_DOCUMENTS}:   {total_r / count:.4f}")
        print(f"F1@{TOP_K_DOCUMENTS}:  {total_f1 / count:.4f}")
    else:
        print("\nNo test questions with gold standard documents found for evaluation.")

    print(f"\nSaving test predictions to {TEST_OUTPUT_JSON}")
    Path("../results").mkdir(parents=True, exist_ok=True)
    with open(TEST_OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({"questions": predictions}, f, indent=2, ensure_ascii=False)

    print("Done")