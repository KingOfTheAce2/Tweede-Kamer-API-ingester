import os
import sys
import json
import time
import logging
import requests
from typing import List, Dict, Optional
from pathlib import Path
from huggingface_hub import HfApi, HfFolder

# ========== Configuration ==========
# You can also override these via environment variables or at the top of the file.
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "100"))
ODATA_URL          = os.getenv("ODATA_URL", "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document")
ODATA_PARAMS       = json.loads(os.getenv("ODATA_PARAMS", '{"$filter": "Verwijderd eq false"}'))
STATE_PATH         = os.getenv("STATE_PATH", "tk_state.json")
OUTPUT_PATH        = os.getenv("OUTPUT_PATH", "tk_crawl.jsonl")
HF_REPO_ID         = os.getenv("HF_REPO_ID", "vGassen/Dutch-Tweede-Kamer-API")
SOURCE_LABEL       = os.getenv("SOURCE_LABEL", "Tweede Kamer")
SHARD_SIZE         = int(os.getenv("SHARD_SIZE", "300")) # ~25 MiB limit
MAX_ENTRIES        = int(os.getenv("MAX_ENTRIES", "10000"))
HF_TOKEN           = os.getenv("HF_TOKEN") or HfFolder.get_token()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== State Management ==========
def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"skip": 0}

def save_state(path: str, state: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)

# ========== Main Crawler ==========
def fetch_documents(skip: int, top: int) -> List[Dict]:
    """Fetch a batch of documents from the OData API."""
    params = ODATA_PARAMS.copy()
    params["$top"] = top
    params["$skip"] = skip
    resp = requests.get(ODATA_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # OData 4.0 returns results in 'value'
    return data.get("value", [])

def clean_text(text):
    """Remove HTML/XML tags and scripts/styles."""
    import re
    if not text:
        return ""
    text = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def emit_jsonl(docs: List[Dict], path: str, label: str):
    """Append to JSONL file, one object per line."""
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs:
            url = doc.get("ResourceUrl") or doc.get("Id") or doc.get("DocumentId") or doc.get("url") or doc.get("@odata.id")
            text = doc.get("Tekst") or doc.get("BodyText") or doc.get("Body") or doc.get("Omschrijving") or doc.get("Titel") or ""
            content = clean_text(text)
            record = {
                "URL": str(url),
                "Content": content,
                "Source": label,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def push_to_hf(jsonl_path: str, repo_id: str, hf_token: str):
    """Push JSONL shards to Hugging Face."""
    api = HfApi()
    repo_files = []
    if not os.path.exists(jsonl_path):
        logging.warning("No output file found.")
        return
    # Split into shards to keep under 25MiB (max 300 records per shard, configurable)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n = 0
    for i in range(0, len(lines), SHARD_SIZE):
        shard_lines = lines[i:i + SHARD_SIZE]
        shard_path = f"shard_{i}_{i+len(shard_lines)}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as sf:
            sf.writelines(shard_lines)
        api.upload_file(
            path_or_fileobj=shard_path,
            path_in_repo=f"shards/{shard_path}",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        repo_files.append(shard_path)
        os.remove(shard_path)
        n += len(shard_lines)
    logging.info(f"Pushed {n} entries in {len(repo_files)} shards to {repo_id}")

def main():
    state = load_state(STATE_PATH)
    skip = state.get("skip", 0)
    total = 0

    while total < MAX_ENTRIES:
        logging.info(f"Fetching documents with $skip={skip}, $top={BATCH_SIZE}")
        for attempt in range(3):
            try:
                docs = fetch_documents(skip, BATCH_SIZE)
                break
            except Exception as e:
                logging.warning(f"Fetch failed (try {attempt+1}/3): {e}")
                time.sleep(3)
        else:
            logging.error("Fetch failed after retries. Aborting.")
            break
        if not docs:
            logging.info("No more documents to fetch.")
            break
        emit_jsonl(docs, OUTPUT_PATH, SOURCE_LABEL)
        total += len(docs)
        skip += len(docs)
        save_state(STATE_PATH, {"skip": skip})
        logging.info(f"Fetched {len(docs)} docs, total={total}")
        if len(docs) < BATCH_SIZE:
            logging.info("Reached last batch.")
            break

    if HF_TOKEN:
        push_to_hf(OUTPUT_PATH, HF_REPO_ID, HF_TOKEN)
    else:
        logging.warning("HF_TOKEN not provided, skipping upload.")

if __name__ == "__main__":
    main()
