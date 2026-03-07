import os
import json
import hashlib
from typing import List, Dict

# Path for our JSON-based memory
MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory_crystal.json")

def _load_memory() -> List[Dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def _save_memory(memory: List[Dict]):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def save_fix_to_memory(repo: str, error_summary: str, broken_file: str, fix_patch: str, issue_category: str = "General"):
    """
    Saves a successful code fix into the JSON Memory Crystal, now with issue categorization.
    """
    memory = _load_memory()
    
    # Check if this exact fix already exists
    fix_id = hashlib.sha256(f"{repo}{error_summary}{fix_patch}{issue_category}".encode()).hexdigest()[:16]
    
    if any(m.get("id") == fix_id for m in memory):
        return fix_id

    memory.append({
        "id": fix_id,
        "repo": repo,
        "error_summary": error_summary,
        "broken_file": broken_file,
        "fix_patch": fix_patch,
        "issue_category": issue_category
    })
    
    _save_memory(memory)
    return fix_id

def query_memory_for_fix(current_error_summary: str, issue_category: str = None, n_results: int = 1):
    """
    Queries the JSON Memory Crystal for similar past errors, prioritizing issue category matches.
    """
    memory = _load_memory()
    if not memory:
        return []
    
    # Simple similarity check: Count overlapping words (case-insensitive)
    def calculate_similarity(s1: str, s2: str):
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0
        intersection = words1.intersection(words2)
        return len(intersection) / max(len(words1), len(words2))

    scored_memory = []
    for m in memory:
        # Boost score if category matches exactly
        category_boost = 0.5 if issue_category and m.get("issue_category") == issue_category else 0.0
        
        base_score = calculate_similarity(current_error_summary, m["error_summary"])
        total_score = base_score + category_boost
        
        if total_score > 0.4:
            scored_memory.append({
                "past_error": m["error_summary"],
                "repo": m["repo"],
                "broken_file": m["broken_file"],
                "fix_patch": m["fix_patch"],
                "category": m.get("issue_category", "N/A"),
                "score": total_score
            })
    
    scored_memory.sort(key=lambda x: x["score"], reverse=True)
    return scored_memory[:n_results]
