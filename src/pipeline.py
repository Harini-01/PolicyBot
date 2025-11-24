# src/full_pipeline.py

import os
import yaml
import logging
from crawler import main as crawl
from downloader import main as download
from preprocess import main as clean
from chunker import main as chunk
#change
import embed as embed_module

#----------- my changes-------------------
import json
import yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
#------------------------------------------

# ----------------------------
# Setup logging
# ----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------------------------
# Track already downloaded / processed files
# ----------------------------
ALREADY_DOWNLOADED_PATH = "data/raw/already_downloaded.yaml"
ALREADY_EMBEDDED_PATH = "data/vector_db/already_embedded.yaml"
ALL_CHUNKS_PATH = "data/chunks/all_chunks.json"

def load_yaml(path):
    if os.path.exists(path):
        with open(path, "r",encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w",encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)

#change
def load_all_chunks():
    if not os.path.exists(ALL_CHUNKS_PATH):
        return []
    with open(ALL_CHUNKS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []
        

def run_pipeline():
    logging.info("Starting full pipeline...")

    # ----------------------------
    # Crawl
    # ----------------------------
    try:
        crawl()
        logging.info("Crawling completed successfully.")
    except Exception as e:
        logging.error(f"Crawling failed: {e}")

    # ----------------------------
    # Download
    # ----------------------------
    already_downloaded = load_yaml(ALREADY_DOWNLOADED_PATH)
    try:
        download()  # downloader.py already handles skipping existing files
        logging.info("Downloading completed successfully.")
    except Exception as e:
        logging.error(f"Downloading failed: {e}")
    finally:
        # update already_downloaded.yaml
        save_yaml(already_downloaded, ALREADY_DOWNLOADED_PATH)

    # ----------------------------
    # Preprocess
    # ----------------------------
    try:
        clean()  # skips non-PDF files, outputs cleaned text
        logging.info("Preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")

    # ----------------------------
    # Chunk
    # ----------------------------
    try:
        chunk()  # generates chunks per file and all_chunks.json
        logging.info("Chunking completed successfully.")
    except Exception as e:
        logging.error(f"Chunking failed: {e}")

    # ----------------------------
    # Embed
    # ----------------------------
     # Load chunks produced by chunker
    all_chunks = load_all_chunks()
    if not all_chunks:
        logging.warning("No chunks found; skipping embedding stage.")
    else:
        # ----------------------------
        # Embed (persistent vector DB + incremental updates)
        # ----------------------------
        already_embedded = load_yaml(ALREADY_EMBEDDED_PATH) or {}
        try:
            # Try to load an existing persisted index + metadata
            index, metadata_list = embed_module.load_index()
            if index is None or not metadata_list:
                # No persisted DB found -> embed all chunks and save
                logging.info("No persisted vector DB found. Embedding all chunks.")
                index, metadata_list = embed_module.embed_all_and_save(all_chunks)
            else:
                # Persisted DB found -> detect new chunks by chunk 'id' and embed only those
                existing_ids = {m.get("id") for m in metadata_list if m.get("id") is not None}
                new_chunks = [c for c in all_chunks if c.get("id") not in existing_ids]
                if not new_chunks:
                    logging.info("No new chunks to embed.")
                else:
                    logging.info(f"Found {len(new_chunks)} new chunks. Embedding incrementally.")
                    index, metadata_list = embed_module.add_embeddings_incremental(index, metadata_list, new_chunks)

            # Update already_embedded tracker (simple count)
            already_embedded = {"count": len(metadata_list)}
            save_yaml(already_embedded, ALREADY_EMBEDDED_PATH)
            logging.info("Embedding stage completed and persisted.")
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            # attempt to persist whatever minimal tracker we have
            try:
                save_yaml(already_embedded, ALREADY_EMBEDDED_PATH)
            except Exception:
                pass

    logging.info("Pipeline completed!")

if __name__ == "__main__":
    run_pipeline()
