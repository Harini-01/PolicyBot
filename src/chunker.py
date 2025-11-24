# ...existing code...
import os
import json
import hashlib
from pathlib import Path
import logging

import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt tokenizer is available (download only if missing)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ----------------------------
# Directories
# ----------------------------
CLEAN_DIR = Path("data/clean")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Setup simple logger
logger = logging.getLogger("chunker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------------
# Helpers
# ----------------------------
def _make_chunk_id(site: str, source_file: str, chunk_index: int, text: str = None) -> str:
    """
    Create a deterministic chunk id based on site, filename stem, and chunk index.
    Optionally include text hash if you want content-derived ids (commented out by default).
    """
    key = f"{site}|{source_file}|{chunk_index}"
    # If you prefer to include chunk content in the id, uncomment:
    # if text is not None:
    #     key += "|" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def chunk_text(text: str, max_words: int = 150):
    """
    Split text into chunks of approx max_words.
    Respects sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words:
            # Finish current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = words.copy()
            current_len = len(words)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ----------------------------
# Main chunking function
# ----------------------------
def main():
    all_chunks = []

    if not CLEAN_DIR.exists():
        logger.warning(f"Clean directory {CLEAN_DIR} does not exist. Nothing to chunk.")
        return

    # iterate in sorted order for determinism
    for site_folder in sorted(CLEAN_DIR.iterdir(), key=lambda p: p.name):
        if not site_folder.is_dir():
            continue

        chunk_site_folder = CHUNK_DIR / site_folder.name
        chunk_site_folder.mkdir(exist_ok=True, parents=True)

        for file_path in sorted(site_folder.iterdir(), key=lambda p: p.name):
            if not file_path.is_file() or file_path.suffix.lower() != ".txt":
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue

            chunks = chunk_text(text)
            chunk_data = []

            out_file = chunk_site_folder / (file_path.stem + "_chunks.json")
            if out_file.exists():
                out_file.unlink()  # remove old chunk file

            for i, chunk in enumerate(chunks):
                chunk_id = _make_chunk_id(site_folder.name, file_path.stem, i)
                chunk_info = {
                    "id": chunk_id,
                    "site": site_folder.name,
                    "source_file": file_path.name,
                    "chunk_index": i,
                    "text": chunk
                }
                chunk_data.append(chunk_info)
                all_chunks.append(chunk_info)

            # Save chunks per file as JSON
            try:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Chunked {file_path} → {out_file}, total chunks: {len(chunks)}")
            except Exception as e:
                logger.error(f"Failed to write chunk file {out_file}: {e}")

    # Save all chunks together for embedding
    all_chunks_file = CHUNK_DIR / "all_chunks.json"
    try:
        with open(all_chunks_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"All chunks saved to {all_chunks_file}, total chunks: {len(all_chunks)}")
    except Exception as e:
        logger.error(f"Failed to write all_chunks file {all_chunks_file}: {e}")

# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    main()
# ...existing code...
