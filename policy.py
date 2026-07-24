import os, json, hashlib, shutil, tempfile, logging, threading
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from emb_pace import PacedEmbeddings

logger = logging.getLogger(__name__)

# In-process cache of the loaded FAISS index, keyed by faiss_dir. Avoids
# re-reading + re-deserializing the index from disk on every single
# search_policy call (previously happened on every policy-tool invocation).
# Invalidated automatically when the on-disk index file's mtime changes
# (i.e. after update_policy_index_if_changed rebuilds it).
_faiss_cache: Dict[str, tuple] = {}
_faiss_cache_lock = threading.Lock()


def _load_faiss_cached(faiss_dir: str, emb: PacedEmbeddings):
    index_file = os.path.join(faiss_dir, "index.faiss")
    if not os.path.exists(index_file):
        return None
    mtime = os.path.getmtime(index_file)
    cached = _faiss_cache.get(faiss_dir)
    if cached and cached[0] == mtime:
        return cached[1]
    with _faiss_cache_lock:
        cached = _faiss_cache.get(faiss_dir)
        if cached and cached[0] == mtime:
            return cached[1]
        vs = FAISS.load_local(faiss_dir, emb, allow_dangerous_deserialization=True)
        _faiss_cache[faiss_dir] = (mtime, vs)
        return vs

@dataclass(frozen=True)
class PolicyConfig:
    faiss_dir: str
    hash_json: str
    blob_conn_str: str
    blob_container: str
    blob_prefix: str = ""                 # e.g. "policies/"
    local_dir: str = "./data/policies"    # 🔧 added: local cache path for PDFs
    throttle_seconds: int = 600
    max_fields_per_doc: int = 40

def update_policy_index_if_changed(cfg: PolicyConfig, splitter, emb: PacedEmbeddings):
    """
    Build/refresh FAISS index for policy PDFs (no RBAC, global access).
    Syncs PDFs from the given blob container/prefix, rebuilds index if new/changed.
    """
    os.makedirs(cfg.faiss_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.hash_json), exist_ok=True)
    os.makedirs(cfg.local_dir, exist_ok=True)

    prev_hash: Dict[str, str] = {}
    if os.path.exists(cfg.hash_json):
        with open(cfg.hash_json, "r", encoding="utf-8") as f:
            prev_hash = json.load(f)

    new_hash: Dict[str, str] = {}
    docs: List[Document] = []

    from azure.storage.blob import ContainerClient
    if cfg.blob_conn_str and cfg.blob_container:
        try:
            cc = ContainerClient.from_connection_string(cfg.blob_conn_str, cfg.blob_container)
            for blob in cc.list_blobs(name_starts_with=cfg.blob_prefix):
                if not blob.name.lower().endswith(".pdf"):
                    continue
                local_path = os.path.join(cfg.local_dir, os.path.basename(blob.name))
                etag = blob.etag.strip('"') if blob.etag else ""
                new_hash[blob.name] = etag
                if prev_hash.get(blob.name) == etag:
                    continue  # unchanged, skip download/index
                # Download updated blob
                with open(local_path, "wb") as f:
                    cc.download_blob(blob.name).readinto(f)
                loader = PyPDFLoader(local_path)
                pdf_docs = loader.load()
                for d in pdf_docs:
                    for i, txt in enumerate(splitter.split_text(d.page_content)):
                        md = dict(d.metadata)
                        md["source"] = blob.name
                        md["chunk_id"] = f"{blob.name}::chunk::{i}"
                        docs.append(Document(page_content=txt, metadata=md))
        except Exception as e:
            logger.error(f"[PolicyIndex] Blob sync failed: {e}")

    if docs:
        vs = FAISS.from_documents(docs, emb)
        tmp = tempfile.mkdtemp(dir=os.path.dirname(cfg.faiss_dir))  # ensure same drive
        vs.save_local(tmp)
        os.makedirs(os.path.dirname(cfg.faiss_dir) or ".", exist_ok=True)
        if os.path.exists(cfg.faiss_dir):
            shutil.rmtree(cfg.faiss_dir, ignore_errors=True)
        os.replace(tmp, cfg.faiss_dir)  # atomic move now works

        logger.info("[PolicyIndex] Rebuilt FAISS index with %d chunks", len(docs))
        with open(cfg.hash_json, "w", encoding="utf-8") as f:
            json.dump(new_hash, f, indent=2)
    else:
        logger.info("[PolicyIndex] No new/changed PDFs found in blob prefix '%s'", cfg.blob_prefix)

def search_policy(cfg: PolicyConfig, emb: PacedEmbeddings, query: str, k: int = 8, query_embedding: Optional[List[float]] = None) -> List[Document]:
    """
    Search the global policy FAISS index (if available).
    """
    if not os.path.exists(cfg.faiss_dir):
        return []
    try:
        vs = _load_faiss_cached(cfg.faiss_dir, emb)
        if vs is None:
            return []
        if query_embedding is not None and hasattr(vs, "similarity_search_by_vector"):
            return vs.similarity_search_by_vector(query_embedding, k=k)
        return vs.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"[PolicySearch] Failed: {e}")
        return []
