"""Clean raw GitHub issues JSON into a simplified, safe structure.

This script is defensive: it handles missing fields, does simple normalization,
and performs exact deduplication of text artifacts (issue bodies and comments)
using SHA256 hashes. Deduplication is registry-based and preserves provenance.
"""

import json
import os
import hashlib
from typing import Any, Dict, List

INPUT_FILE = "/Users/luffy_sama/Desktop/Workspace/Try/10/data/raw/airflow_issues.json"
OUTPUT_FILE = "/Users/luffy_sama/Desktop/Workspace/Try/10/data/processed/airflow_clean.json"

# Global exact-hash registry used to deduplicate text artifacts.
# Maps sha256(normalized_text) -> artifact_id (issue number or comment id)
artifact_hash_registry: Dict[str, Any] = {}


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def normalize_text(text: Any) -> str:
    if not text:
        return ""
    t = str(text).lower()
    t = " ".join(t.split())
    return t


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_comment(c: Dict[str, Any]) -> Dict[str, Any]:
    raw = c.get("body") or ""
    normalized = normalize_text(raw)
    h = hash_text(normalized)

    if h in artifact_hash_registry:
        duplicate_of = artifact_hash_registry[h]
    else:
        artifact_hash_registry[h] = c.get("id")
        duplicate_of = None

    return {
        "comment_id": c.get("id"),
        "author": _safe_get(c, "user", "login", default=None),
        "body": raw,
        "normalized_body": normalized,
        "normalized_hash": h,
        "is_duplicate_of": duplicate_of,
        "created_at": c.get("created_at"),
    }


def clean_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    labels = issue.get("labels") or []
    comments = issue.get("comments_data") or []

    raw_body = issue.get("body") or ""
    normalized_body = normalize_text(raw_body)
    body_hash = hash_text(normalized_body)

    if body_hash in artifact_hash_registry:
        body_duplicate_of = artifact_hash_registry[body_hash]
    else:
        artifact_hash_registry[body_hash] = issue.get("number")
        body_duplicate_of = None

    return {
        "issue_id": issue.get("number"),
        "title": issue.get("title") or "",
        "body": raw_body,
        "normalized_body": normalized_body,
        "normalized_hash": body_hash,
        "is_duplicate_of": body_duplicate_of,
        "author": _safe_get(issue, "user", "login", default=None),
        "labels": [l.get("name") for l in labels if isinstance(l, dict)],
        "state": issue.get("state") or "",
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "comments": [clean_comment(c) for c in comments if isinstance(c, dict)],
    }


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from {INPUT_FILE}: {e}")
        return
    except Exception as e:
        print(f"Error reading {INPUT_FILE}: {e}")
        return

    if not isinstance(data, list):
        print(f"Unexpected data format in {INPUT_FILE}; expected a list of issues.")
        return

    cleaned: List[Dict[str, Any]] = [clean_issue(i) for i in data]

    out_dir = os.path.dirname(OUTPUT_FILE) or "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write output file {OUTPUT_FILE}: {e}")
        return

    print(f"Preprocessing complete. Wrote {len(cleaned)} issues to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()