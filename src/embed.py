# ...existing code...
import os
import json
import yaml
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configuration
VECTOR_DB_DIR = "data/vector_db"
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.json")
ALREADY_EMBEDDED_PATH = os.path.join(VECTOR_DB_DIR, "already_embedded.yaml")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # change if needed

# Module-level cached model
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def _ensure_dir():
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

def compute_embeddings(texts):
    """
    Compute embeddings for a list of texts and L2-normalize them (for IP similarity).
    Returns a numpy array shape (n, d).
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    model = _get_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # Ensure float32 and normalize for inner-product search
    embs = embs.astype("float32")
    faiss.normalize_L2(embs)
    return embs

def build_faiss_index(dimension):
    """
    Build a FAISS IndexIDMap over IndexFlatIP so vectors can be added with deterministic ids and persisted.
    """
    base = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base)
    return index

def save_index(index, metadata_list):
    """
    Persist the FAISS index and metadata to disk.
    """
    _ensure_dir()
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
    except Exception as e:
        logging.error(f"Failed to write faiss index: {e}")
        raise
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    with open(ALREADY_EMBEDDED_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"count": len(metadata_list)}, f)

def load_index():
    """
    Load persisted FAISS index and metadata. Returns (index, metadata_list) or (None, []) if not present.
    """
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        return None, []
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        logging.error(f"Failed to read faiss index: {e}")
        return None, []
    # ensure we have an IndexIDMap so add_with_ids works predictably
    try:
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
    except Exception:
        # if the type check fails for some faiss build, continue with the loaded index
        pass
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read metadata.json: {e}")
        metadata_list = []
    return index, metadata_list

def embed_all_and_save(all_chunks):
    """
    Embed all chunks from scratch, build a new index and metadata file, persist them, and return (index, metadata_list).
    all_chunks: list of dicts containing at least 'id' and 'text'.
    """
    texts = [c.get("text", "") for c in all_chunks]
    embeddings = compute_embeddings(texts)
    if embeddings.size == 0:
        # empty index
        index = build_faiss_index(1)
        metadata_list = []
        save_index(index, metadata_list)
        return index, metadata_list

    d = embeddings.shape[1]
    index = build_faiss_index(d)
    ids = np.arange(0, embeddings.shape[0], dtype="int64")
    index.add_with_ids(embeddings, ids)

    metadata_list = []
    for c in all_chunks:
        md = {
            "id": c.get("id"),
            "site": c.get("site"),
            "source_file": c.get("source_file"),
            "chunk_index": c.get("chunk_index"),
        }
        metadata_list.append(md)

    save_index(index, metadata_list)
    return index, metadata_list

def add_embeddings_incremental(index, metadata_list, new_chunks):
    """
    Compute embeddings for new_chunks and append them to the provided index and metadata_list.
    Returns updated (index, metadata_list).
    """
    if not new_chunks:
        return index, metadata_list

    texts = [c.get("text", "") for c in new_chunks]
    embeddings = compute_embeddings(texts)
    if embeddings.size == 0:
        return index, metadata_list

    # Verify dimensionality matches the index
    try:
        index_dim = index.d
    except Exception:
        index_dim = None
    if index_dim is not None and embeddings.shape[1] != index_dim:
        raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) does not match FAISS index dimension ({index_dim}).")

    start_id = len(metadata_list)
    n = embeddings.shape[0]
    ids = np.arange(start_id, start_id + n, dtype="int64")

    # Try to add with ids (works when index is IndexIDMap)
    try:
        index.add_with_ids(embeddings, ids)
    except Exception:
        # fallback to add (ids will be implicit)
        index.add(embeddings)

    for c in new_chunks:
        md = {
            "id": c.get("id"),
            "site": c.get("site"),
            "source_file": c.get("source_file"),
            "chunk_index": c.get("chunk_index"),
        }
        metadata_list.append(md)

    save_index(index, metadata_list)
    return index, metadata_list

# Optional helper: simple search function using the saved index and metadata
def search(index, metadata_list, query, top_k=5):
    """
    Return top_k metadata entries similar to query. Returns list of (metadata, score).
    """
    if index is None or not metadata_list:
        return []
    q_emb = compute_embeddings([query])
    if q_emb.size == 0:
        return []
    D, I = index.search(q_emb, top_k)  # distances and indices
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata_list):
            continue
        results.append((metadata_list[int(idx)], float(dist)))
    return results
# ...existing code...
