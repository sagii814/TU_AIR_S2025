import json
import os
from pathlib import Path
from tqdm import tqdm
from data_fetching import DataFetcher


# Load BioASQ questions
def load_bioasq_questions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']


# Extract all unique PubMed IDs
def get_unique_pubmed_ids(questions):
    pubmed_ids = set()
    for q in questions:
        for doc_url in q.get("documents", []):
            if doc_url and "pubmed" in doc_url:
                try:
                    pubmed_id = doc_url.strip().split("/")[-1]
                    if pubmed_id.isdigit():  # makes sure it's a real PubMed ID
                        pubmed_ids.add(pubmed_id)
                except Exception:
                    continue
    return list(pubmed_ids)


# Fetch abstracts and cache to JSON
def fetch_and_cache_abstracts(pubmed_ids, fetcher, cache_path):
    if Path(cache_path).exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
    else:
        cached = {}

    to_fetch = [pid for pid in pubmed_ids if pid not in cached]

    print(f"Fetching {len(to_fetch)} new abstracts...")
    for pid in tqdm(to_fetch):
        try:
            results = fetcher.fetch_data(query=pid, max_results=1, keys={"pubmed_id", "abstract"})
            if results and results[0].get("abstract"):
                cached[pid] = results[0]["abstract"]
        except Exception as e:
            print(f"Error fetching {pid}: {e}")

    # Save updated cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cached, f, indent=2)

    return cached


if __name__ == "__main__":
    # Inputs
    BIOASQ_JSON = "../data/test.json"
    CACHE_FILE = "../data/pubmed_abstracts_cache.json"
    EMAIL = "your@email.com"
    TOOL = "BM25Pipeline"

    # Load and parse
    questions = load_bioasq_questions(BIOASQ_JSON)
    pubmed_ids = get_unique_pubmed_ids(questions)

    # Fetch abstracts
    fetcher = DataFetcher(tool=TOOL, email=EMAIL)
    abstracts_dict = fetch_and_cache_abstracts(pubmed_ids, fetcher, CACHE_FILE)

    print(f"Cached {len(abstracts_dict)} abstracts")