"""
Memory Crystal — the agent's long-term memory of past fixes (RAG).

Primary backend: ChromaDB, a real persistent vector store. Every fix is embedded
(all-MiniLM-L6-v2 by default) so retrieval is true semantic similarity, not word overlap.

If ChromaDB can't be imported/initialised (e.g. a stripped-down demo box), we fall back
to the legacy JSON + word-overlap store so the pipeline still runs. Both backends expose
the same two functions with identical return shapes, so callers never change.
"""
import os
import json
import hashlib
from typing import List, Dict, Optional

_STORE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(_STORE_DIR, "memory_crystal_db")
MEMORY_FILE = os.path.join(_STORE_DIR, "memory_crystal.json")  # legacy fallback

# ------------------------------------------------------------------ #
#  Try to stand up the real vector DB. Fall back silently on failure. #
# ------------------------------------------------------------------ #
_collection = None
_USING_VECTOR_DB = False
try:
    import chromadb

    _client = chromadb.PersistentClient(path=CHROMA_DIR)
    # cosine space so similarity = 1 - distance is well-behaved in [0, 2]
    _collection = _client.get_or_create_collection(
        name="fixes", metadata={"hnsw:space": "cosine"}
    )
    _USING_VECTOR_DB = True
    print(f"[MEMORY] Vector DB (ChromaDB) online at {CHROMA_DIR}")
except Exception as e:  # noqa: BLE001 — any failure => JSON fallback
    print(f"[MEMORY] ChromaDB unavailable ({e}); using JSON fallback store.")


def _fix_id(repo: str, error_summary: str, fix_patch: str, issue_category: str) -> str:
    raw = f"{repo}{error_summary}{fix_patch}{issue_category}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ------------------------------------------------------------------ #
#  Public API                                                        #
# ------------------------------------------------------------------ #
def save_fix_to_memory(
    repo: str,
    error_summary: str,
    broken_file: str,
    fix_patch: str,
    issue_category: str = "General",
) -> str:
    """Persist a successful fix. Returns a stable content hash id (idempotent)."""
    fid = _fix_id(repo, error_summary, fix_patch, issue_category)

    if _USING_VECTOR_DB:
        try:
            # upsert => calling twice with the same fix is a no-op, no duplicates
            _collection.upsert(
                ids=[fid],
                documents=[error_summary],  # embedded => semantic search key
                metadatas=[{
                    "repo": repo,
                    "broken_file": broken_file,
                    "fix_patch": fix_patch,
                    "issue_category": issue_category,
                    "error_summary": error_summary,
                }],
            )
            return fid
        except Exception as e:  # noqa: BLE001
            print(f"[MEMORY] Vector upsert failed ({e}); writing to JSON fallback.")

    return _json_save(fid, repo, error_summary, broken_file, fix_patch, issue_category)


def query_memory_for_fix(
    current_error_summary: str,
    issue_category: Optional[str] = None,
    n_results: int = 1,
) -> List[Dict]:
    """
    Return up to n_results past fixes most similar to the current error.
    Same-category matches get a similarity boost. Return shape is stable across backends:
        {past_error, repo, broken_file, fix_patch, category, score}
    """
    if _USING_VECTOR_DB:
        try:
            return _vector_query(current_error_summary, issue_category, n_results)
        except Exception as e:  # noqa: BLE001
            print(f"[MEMORY] Vector query failed ({e}); falling back to JSON.")

    return _json_query(current_error_summary, issue_category, n_results)


# ------------------------------------------------------------------ #
#  Vector backend                                                    #
# ------------------------------------------------------------------ #
def _vector_query(error_summary: str, issue_category, n_results: int) -> List[Dict]:
    if _collection.count() == 0:
        return []

    # over-fetch, then re-rank with the category boost in Python
    fetch_k = min(max(n_results * 4, 5), _collection.count())
    res = _collection.query(query_texts=[error_summary], n_results=fetch_k)

    metadatas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    scored = []
    for meta, dist in zip(metadatas, distances):
        base = 1.0 - float(dist)  # cosine distance -> similarity
        boost = 0.15 if issue_category and meta.get("issue_category") == issue_category else 0.0
        total = base + boost
        if total < 0.35:  # weak semantic match — skip
            continue
        scored.append({
            "past_error": meta.get("error_summary", ""),
            "repo": meta.get("repo", ""),
            "broken_file": meta.get("broken_file", ""),
            "fix_patch": meta.get("fix_patch", ""),
            "category": meta.get("issue_category", "N/A"),
            "score": round(total, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_results]


# ------------------------------------------------------------------ #
#  Legacy JSON backend (fallback)                                    #
# ------------------------------------------------------------------ #
def _load_memory() -> List[Dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return []


def _json_save(fid, repo, error_summary, broken_file, fix_patch, issue_category) -> str:
    memory = _load_memory()
    if any(m.get("id") == fid for m in memory):
        return fid
    memory.append({
        "id": fid,
        "repo": repo,
        "error_summary": error_summary,
        "broken_file": broken_file,
        "fix_patch": fix_patch,
        "issue_category": issue_category,
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
    return fid


def _json_query(error_summary: str, issue_category, n_results: int) -> List[Dict]:
    memory = _load_memory()
    if not memory:
        return []

    def similarity(s1: str, s2: str) -> float:
        w1, w2 = set(s1.lower().split()), set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / max(len(w1), len(w2))

    scored = []
    for m in memory:
        boost = 0.5 if issue_category and m.get("issue_category") == issue_category else 0.0
        total = similarity(error_summary, m["error_summary"]) + boost
        if total > 0.4:
            scored.append({
                "past_error": m["error_summary"],
                "repo": m["repo"],
                "broken_file": m["broken_file"],
                "fix_patch": m["fix_patch"],
                "category": m.get("issue_category", "N/A"),
                "score": total,
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_results]
