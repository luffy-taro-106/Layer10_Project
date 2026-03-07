import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

CLAIMS_FILE = "data/claims/claims_v1.json"
OUTPUT_FILE = "data/Entities/entities_v1.json"

def get_entity_type(entity_id):
    if entity_id.startswith("issue_"):
        return "Issue"
    elif entity_id.startswith("comment_"):
        return "Comment"
    elif entity_id.startswith("person_"):
        return "Person"
    elif entity_id.startswith("label_"):
        return "Label"
    elif entity_id.startswith("status_"):
        return "Status"
    else:
        return "Unknown"


def _canonicalize_person(raw: str) -> str:
    token = raw.replace("person_", "")
    token = token.lower()
    token = token.replace("-", "")
    token = re.sub(r"\s+", "", token)
    return f"person_{token}"


def _canonicalize_label(raw: str) -> str:
    token = raw.replace("label_", "")
    token = token.lower()
    token = token.replace(":", "_")
    token = token.replace("-", "_")
    token = token.replace("__", "_")
    token = token.strip("_")
    return f"label_{token}"


def _canonicalize_status(raw: str) -> str:
    token = raw.replace("status_", "")
    token = token.lower()
    token = token.replace("-", "_")
    return f"status_{token}"


def canonicalize_entity_id(entity_id: str) -> str:
    if not isinstance(entity_id, str):
        return entity_id
    if entity_id.startswith("person_"):
        return _canonicalize_person(entity_id)
    if entity_id.startswith("label_"):
        return _canonicalize_label(entity_id)
    if entity_id.startswith("status_"):
        return _canonicalize_status(entity_id)
    # leave issues and others as-is
    return entity_id

def main():
    with open(CLAIMS_FILE) as f:
        claims = json.load(f)

    entities: Dict[str, Dict[str, Any]] = {}
    merge_log: List[Dict[str, Any]] = []

    for claim in claims:
        event_time = claim.get("event_time")
        for node in (claim.get("subject"), claim.get("object")):
            if node is None:
                continue
            canonical = canonicalize_entity_id(node)
            entity_type = get_entity_type(canonical)
            if entity_type == "Unknown":
                continue

            # ensure canonical entity exists
            ent = entities.get(canonical)
            if ent is None:
                ent = {
                    "entity_id": canonical,
                    "entity_type": entity_type,
                    "created_at": event_time,
                    "last_seen_at": event_time,
                    "entity_version": "1.0",
                    "aliases": [],
                }
                entities[canonical] = ent

            # If original node differs from canonical, record alias and merge
            if node != canonical:
                if node not in ent["aliases"]:
                    ent["aliases"].append(node)
                    # record merge operation for reversibility
                    merge_log.append({
                        "merge_id": str(uuid.uuid4()),
                        "source_entity": node,
                        "target_entity": canonical,
                        "reason": "deterministic_normalization",
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    })

            # update created_at (earliest) and last_seen_at (latest)
            if event_time:
                if not ent.get("created_at") or (event_time < ent.get("created_at")):
                    ent["created_at"] = event_time
                if not ent.get("last_seen_at") or (event_time > ent.get("last_seen_at")):
                    ent["last_seen_at"] = event_time

    os.makedirs(os.path.dirname(OUTPUT_FILE) or "data/graph", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(entities.values()), f, indent=2, ensure_ascii=False)

    # write merge log for reversibility/audit
    merge_log_path = os.path.join(os.path.dirname(OUTPUT_FILE) or "data/graph", "merge_log.json")
    try:
        with open(merge_log_path, "w", encoding="utf-8") as mf:
            json.dump(merge_log, mf, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print(f"Generated {len(entities)} entities.")

if __name__ == "__main__":
    main()
