import sqlite3
import requests
import subprocess
import os
import logging
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi
from typing import List, Dict, Optional, Tuple

# --- Configuration ---
# The total number of documents to process in a single run.
TOTAL_DOCUMENT_LIMIT = 1000
# The number of documents to include in each upload batch to Hugging Face.
UPLOAD_BATCH_SIZE = 200
# The local SQLite database to store the skiptoken for resuming progress.
DB_PATH = "progress.sqlite3"
# The file to store detailed logs.
LOG_FILE = "ingestion_log.log"
# The base URL for the Tweede Kamer API feed.
API_BASE_URL = "https://gegevensmagazijn.tweedekamer.nl/SyncFeed/2.0/Feed"
# The category of documents to fetch.
API_CATEGORY = "Document"

# --- XML Namespaces ---
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
TK_NAMESPACE = "http://www.tweedekamer.nl/xsd/tkData/v1-0"
NAMESPACES = {'atom': ATOM_NAMESPACE, 'tk': TK_NAMESPACE}

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def setup_database():
    """Ensures the progress tracking database and table exist."""
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS progress(category TEXT PRIMARY KEY, skiptoken INTEGER)"
            )
            logging.info(f"Database '{DB_PATH}' setup complete.")
    except sqlite3.Error as e:
        logging.error(f"Database setup failed: {e}")
        raise

def get_skiptoken(category: str) -> int:
    """Retrieves the last saved skiptoken for a given category."""
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.execute("SELECT skiptoken FROM progress WHERE category=?", (category,))
            row = cur.fetchone()
            if row:
                logging.info(f"Resuming from skiptoken: {row[0]} for category '{category}'.")
                return row[0]
            else:
                logging.info(f"No skiptoken found for category '{category}'. Starting from the beginning.")
                return -1
    except sqlite3.Error as e:
        logging.error(f"Failed to get skiptoken: {e}")
        return -1

def save_skiptoken(category: str, skiptoken: int) -> None:
    """Saves the latest skiptoken for a category to the database."""
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("REPLACE INTO progress(category, skiptoken) VALUES(?,?)", (category, skiptoken))
            con.commit()
            logging.info(f"Saved skiptoken: {skiptoken} for category '{category}'.")
    except sqlite3.Error as e:
        logging.error(f"Failed to save skiptoken {skiptoken}: {e}")

def convert_pdf_to_text(pdf_content: bytes) -> str:
    """Converts PDF content from bytes to plain text using the 'pdftotext' utility."""
    try:
        process = subprocess.run(
            ["pdftotext", "-q", "-", "-"],
            input=pdf_content,
            capture_output=True,
            check=True,
        )
        return process.stdout.decode('utf-8', errors='ignore')
    except FileNotFoundError:
        logging.error("`pdftotext` utility not found. Please install poppler-utils.")
        return ""
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "No stderr."
        logging.error(f"pdftotext failed with exit code {e.returncode}: {error_output}")
        return ""
    except Exception as e:
        logging.error(f"An unexpected error occurred during PDF conversion: {e}")
        return ""

def fetch_and_process_entry(entry: etree._Element) -> Optional[Dict[str, str]]:
    """
    Parses a single <entry> element, downloads its content, and returns a structured dict.
    This function also handles scraping of XML/HTML content by returning it as plain text.
    """
    entry_id = entry.find("atom:id", NAMESPACES).text if entry.find("atom:id", NAMESPACES) is not None else "N/A"
    
    # Check if the entry is marked as deleted
    content_element = entry.find("atom:content", NAMESPACES)
    if content_element is not None and content_element.text:
        try:
            nested_xml = etree.fromstring(content_element.text.encode('utf-8'))
            if nested_xml.get(f"{{{TK_NAMESPACE}}}verwijderd") == "true":
                logging.info(f"Skipping entry {entry_id}: marked as deleted.")
                return None
        except etree.XMLSyntaxError:
            pass # Not all content is XML

    # Find the enclosure link which contains the actual document
    enclosure_link = entry.find("atom:link[@rel='enclosure']", NAMESPACES)
    if enclosure_link is None or not enclosure_link.get("href"):
        logging.warning(f"Skipping entry {entry_id}: no enclosure URL found.")
        return None
    
    enclosure_url = enclosure_link.get("href")
    
    try:
        # Download the document content
        dresp = requests.get(enclosure_url, timeout=60)
        dresp.raise_for_status()
        content_type = dresp.headers.get('Content-Type', '').split(';')[0].strip().lower()
        
        fetched_content = ""
        if content_type == "application/pdf":
            logging.info(f"Converting PDF: {enclosure_url}")
            fetched_content = convert_pdf_to_text(dresp.content)
        elif content_type.startswith("text/") or content_type == "application/xml":
            logging.info(f"Scraping text/xml content from: {enclosure_url}")
            # For text or XML, we just use the text content directly.
            # This effectively "scrapes" the content from these files.
            fetched_content = dresp.text
        else:
            logging.warning(f"Skipping unsupported content type '{content_type}' for URL: {enclosure_url}")
            return None
            
        if not fetched_content.strip():
            logging.warning(f"Content from {enclosure_url} is empty after processing. Skipping.")
            return None
            
        return {"URL": enclosure_url, "content": fetched_content, "Source": "Tweede Kamer"}

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching enclosure '{enclosure_url}': {e}")
        return None

def fetch_api_page(category: str, skiptoken: Optional[int]) -> Tuple[List[Dict[str, str]], Optional[int]]:
    """Fetches one page of results from the API and returns documents and the next skiptoken."""
    params = {"category": category}
    if skiptoken is not None and skiptoken > 0:
        params["skiptoken"] = skiptoken
    
    logging.info(f"Fetching API with params: {params}")
    try:
        resp = requests.get(API_BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch API feed: {e}")
        return [], None

    root = etree.fromstring(resp.content)
    entries = root.findall("atom:entry", NAMESPACES)
    logging.info(f"API returned {len(entries)} entries.")
    
    processed_docs = []
    for entry in entries:
        doc = fetch_and_process_entry(entry)
        if doc:
            processed_docs.append(doc)

    # Find the skiptoken for the next page
    next_link = root.find("atom:link[@rel='next']", NAMESPACES)
    next_skiptoken = None
    if next_link is not None and "skiptoken=" in next_link.get("href", ""):
        try:
            token_str = next_link.get("href").split("skiptoken=")[1].split("&")[0]
            next_skiptoken = int(token_str)
        except (ValueError, IndexError):
            logging.warning("Could not parse skiptoken from 'next' link.")
            
    return processed_docs, next_skiptoken

def push_batch_to_hf(docs: List[Dict[str, str]], repo_id: str, batch_number: int):
    """Pushes a list of documents as a Parquet file to a Hugging Face dataset repo."""
    if not docs:
        logging.info("No documents in the current batch to push.")
        return
        
    logging.info(f"--- Preparing to push batch #{batch_number} with {len(docs)} documents to {repo_id} ---")
    api = HfApi()
    local_parquet_path = f"data_batch_{batch_number}.parquet"
    repo_file_path = f"data/batch_{batch_number}.parquet"
    
    try:
        # Create a Dataset object and save to a local Parquet file
        ds = Dataset.from_list(docs)
        ds.to_parquet(local_parquet_path)
        logging.info(f"Successfully saved batch to '{local_parquet_path}'.")

        # Upload the file to the Hugging Face Hub
        api.upload_file(
            path_or_fileobj=local_parquet_path,
            path_in_repo=repo_file_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logging.info(f"Successfully uploaded batch #{batch_number} to repository.")
        
    except Exception as e:
        logging.error(f"Failed to upload batch to Hugging Face: {e}")
    finally:
        # Clean up the local file after upload
        if os.path.exists(local_parquet_path):
            os.remove(local_parquet_path)
            logging.info(f"Cleaned up local file: {local_parquet_path}")

def main():
    """Main function to run the ingestion and upload process."""
    logging.info("--- Starting ingestion process ---")
    
    # Ensure HF_REPO_ID is set, e.g., "vGassen/Dutch-Tweede-Kamer-API"
    hf_repo_id = os.getenv("HF_REPO_ID")
    if not hf_repo_id:
        logging.error("HF_REPO_ID environment variable not set. Please set it to your Hugging Face repo ID.")
        return

    setup_database()
    
    total_docs_collected = 0
    batch_number = 1
    docs_for_current_batch = []
    
    # Get the starting skiptoken from the last run
    current_skiptoken = get_skiptoken(API_CATEGORY)

    while total_docs_collected < TOTAL_DOCUMENT_LIMIT:
        remaining_limit = TOTAL_DOCUMENT_LIMIT - total_docs_collected
        
        # Fetch a page of data from the API
        new_docs, next_skiptoken = fetch_api_page(API_CATEGORY, current_skiptoken)
        
        if not new_docs and next_skiptoken is None:
            logging.info("No more documents available from the feed. Ending process.")
            break
        
        # Add new documents to the batch, respecting the total limit
        docs_to_add = new_docs[:remaining_limit]
        docs_for_current_batch.extend(docs_to_add)
        total_docs_collected += len(docs_to_add)
        
        logging.info(f"Collected {len(docs_to_add)} new documents. Total collected: {total_docs_collected}/{TOTAL_DOCUMENT_LIMIT}")

        # If batch is full, upload it
        if len(docs_for_current_batch) >= UPLOAD_BATCH_SIZE:
            push_batch_to_hf(docs_for_current_batch, hf_repo_id, batch_number)
            # Save progress *after* a successful push
            if current_skiptoken is not None:
                save_skiptoken(API_CATEGORY, current_skiptoken)
            batch_number += 1
            docs_for_current_batch = [] # Reset for the next batch
        
        # Update skiptoken and check for end of feed
        current_skiptoken = next_skiptoken
        if current_skiptoken is None:
            logging.info("Reached the end of the API feed.")
            break

    # Push any remaining documents in the last batch
    if docs_for_current_batch:
        logging.info("Pushing the final batch of documents.")
        push_batch_to_hf(docs_for_current_batch, hf_repo_id, batch_number)
    
    # Save the final skiptoken
    if current_skiptoken is not None:
        save_skiptoken(API_CATEGORY, current_skiptoken)
        
    logging.info(f"--- Ingestion process finished. Collected a total of {total_docs_collected} documents. ---")

if __name__ == "__main__":
    main()
