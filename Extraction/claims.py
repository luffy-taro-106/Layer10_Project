"""Generate deterministic claims from processed issue metadata.

Claim IDs are deterministic using UUIDv5 over a stable string composed
from subject, predicate, object, event_time and source_id.
"""

import json
import os
import uuid
import re
from typing import Any, Dict, List
from datetime import datetime

INPUT_FILE = "/Users/luffy_sama/Desktop/Workspace/Try/10/data/processed/airflow_clean.json"
OUTPUT_FILE = "/Users/luffy_sama/Desktop/Workspace/Try/10/data/claims/claims_v1.json"
EXCERPT_LIMIT = 300


def _stable_claim_id(subject: str, predicate: str, obj: str, event_time: str, source_id: Any) -> str:
    key = "|".join([str(subject), str(predicate), str(obj), str(event_time), str(source_id)])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _sanitize_token(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.strip()
    s = s.lower()
    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = s.replace("\"", "")
    s = s.replace("'", "")
    s = s.replace(":", "_")
    s = s.replace("-", "_")
    # collapse multiple underscores into one
    while "__" in s:
        s = s.replace("__", "_")
    return s


def namespace_object(predicate: str, obj: Any) -> str:
    token = _sanitize_token(obj)
    # person-like predicates
    if predicate in ("ISSUE_CREATED_BY", "COMMENT_WRITTEN_BY", "ISSUE_ASSIGNED_TO"):
        return f"person_{token}"
    if predicate == "ISSUE_HAS_LABEL":
        # If the label already encodes a category (e.g. "kind_bug" or "area_core"),
        # avoid producing "label_kind_kind_bug" and instead produce "label_kind_bug"
        categories = ("kind_", "area_", "priority_", "component_")
        if any(token.startswith(cat) for cat in categories):
            return f"label_{token}"
        return f"label_kind_{token}"
    if predicate == "ISSUE_HAS_STATUS":
        return f"status_{token}"
    return token


def _find_offsets(source_text: Any, excerpt: str):
    if not source_text or not excerpt:
        return {"char_start": None, "char_end": None}
    try:
        s = str(source_text)
        ex = str(excerpt)
        # try exact match
        idx = s.find(ex)
        if idx != -1:
            return {"char_start": idx, "char_end": idx + len(ex)}

        # try normalized whitespace match
        import re

        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x).strip()

        ns = norm(s)
        ne = norm(ex)
        idx = ns.find(ne)
        if idx != -1:
            # approximate offset in original text: find the first token
            token = ne.split()[0]
            approx_idx = s.find(token)
            if approx_idx != -1:
                return {"char_start": approx_idx, "char_end": approx_idx + len(ne)}
    except Exception:
        pass
    return {"char_start": None, "char_end": None}


def make_claim(subject: str, predicate: str, obj: Any, event_time: str, source_type: str, source_id: Any, excerpt: str, source_text: Any = None, location: str | None = None) -> Dict[str, Any]:
    excerpt_text = (excerpt or "")[:EXCERPT_LIMIT]
    claim_id = _stable_claim_id(subject, predicate, obj, event_time, source_id)
    offsets = _find_offsets(source_text, excerpt_text)
    extraction_ts = datetime.utcnow().isoformat() + "Z"
    return {
        "claim_id": claim_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "event_time": event_time,
        "valid_from": event_time,
        "valid_to": None,
        "confidence": 1.0,
        "extraction_metadata": {
            "schema_version": "1.0",
            "extractor_version": "deterministic_v1",
            "model": None,
            "prompt_version": None,
            "extracted_at": extraction_ts,
        },
        "evidence": {
            "source_type": source_type,
            "source_id": source_id,
            "excerpt": excerpt_text,
            "location": location,
            "offsets": offsets,
            "timestamp": event_time,
            "extraction": {"method": source_type, "timestamp": extraction_ts},
        },
    }


def _load_schema(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _basic_validate_claim(claim: Dict[str, Any]) -> bool:
    # Basic presence/type checks
    required = ["claim_id", "subject", "predicate", "object", "event_time", "valid_from", "confidence", "evidence", "extraction_metadata"]
    for r in required:
        if r not in claim:
            return False

    if not isinstance(claim["confidence"], (int, float)):
        return False

    ev = claim.get("evidence") or {}
    if not isinstance(ev, dict):
        return False

    em = claim.get("extraction_metadata") or {}
    if not isinstance(em, dict):
        return False
    # minimal extraction metadata checks
    if "schema_version" not in em or "extracted_at" not in em:
        return False

    return True


def validate_claim(claim: Dict[str, Any], schema_path: str = "data/claims/schema_v1.json") -> bool:
    schema = _load_schema(schema_path)
    if not schema:
        # fallback to basic validation if schema not available
        return _basic_validate_claim(claim)

    try:
        import jsonschema # type: ignore
        jsonschema.validate(instance=claim, schema=schema)
        return True
    except Exception:
        # fallback to basic
        return _basic_validate_claim(claim)


def is_subject_allowed(predicate: str, subject: str) -> bool:
    # mapping predicates to allowed subject prefixes
    mapping = {
        "ISSUE_CREATED_BY": ("issue_",),
        "ISSUE_HAS_STATUS": ("issue_",),
        "ISSUE_HAS_LABEL": ("issue_",),
        "ISSUE_HAS_COMMENT": ("issue_",),
        "COMMENT_WRITTEN_BY": ("comment_",),
        "ISSUE_ASSIGNED_TO": ("issue_",)
    }
    allowed = mapping.get(predicate)
    if not allowed:
        return True
    return any(subject.startswith(p) for p in allowed)


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            issues = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from {INPUT_FILE}: {e}")
        return
    except Exception as e:
        print(f"Error reading {INPUT_FILE}: {e}")
        return

    if not isinstance(issues, list):
        print(f"Unexpected data format in {INPUT_FILE}; expected a list of issues.")
        return

    claims: List[Dict[str, Any]] = []

    for issue in issues:
        issue_id = issue.get("issue_id") or issue.get("number")
        issue_node = f"issue_{issue_id}"
        event_time = issue.get("created_at") or issue.get("updated_at") or ""

        # ISSUE_CREATED_BY
        author = issue.get("author")
        if author:
            obj_ns = namespace_object("ISSUE_CREATED_BY", author)
            c = make_claim(
                subject=issue_node,
                predicate="ISSUE_CREATED_BY",
                obj=obj_ns,
                event_time=event_time,
                source_type="issue_metadata",
                source_id=issue_id,
                excerpt=issue.get("title") or "",
            )
            if is_subject_allowed(c["predicate"], c["subject"]) and validate_claim(c):
                claims.append(c)
            else:
                print(f"Skipped invalid or disallowed claim: {c.get('claim_id')}")

        # ISSUE_HAS_STATUS
        state = issue.get("state")
        if state:
            obj_ns = namespace_object("ISSUE_HAS_STATUS", state)
            c = make_claim(
                subject=issue_node,
                predicate="ISSUE_HAS_STATUS",
                obj=obj_ns,
                event_time=issue.get("updated_at") or event_time,
                source_type="issue_metadata",
                source_id=issue_id,
                excerpt=f"State: {state}",
            )
            if is_subject_allowed(c["predicate"], c["subject"]) and validate_claim(c):
                claims.append(c)
            else:
                print(f"Skipped invalid or disallowed claim: {c.get('claim_id')}")

        # ISSUE_HAS_LABEL
        for label in (issue.get("labels") or []):
            obj_ns = namespace_object("ISSUE_HAS_LABEL", label)
            c = make_claim(
                subject=issue_node,
                predicate="ISSUE_HAS_LABEL",
                obj=obj_ns,
                event_time=event_time,
                source_type="issue_metadata",
                source_id=issue_id,
                excerpt=f"Label: {label}",
            )
            if is_subject_allowed(c["predicate"], c["subject"]) and validate_claim(c):
                claims.append(c)
            else:
                print(f"Skipped invalid or disallowed claim: {c.get('claim_id')}")

    # ISSUE_ASSIGNED_TO from issue-level assignees
    assignees = issue.get("assignees") or []
    if assignees and not isinstance(assignees, (list, tuple)):
        assignees = [assignees]
    for assignee in assignees:
        if not assignee:
            continue
        obj_ns = namespace_object("ISSUE_ASSIGNED_TO", assignee)
        c = make_claim(
            subject=issue_node,
            predicate="ISSUE_ASSIGNED_TO",
            obj=obj_ns,
            event_time=event_time,
            source_type="issue_metadata",
            source_id=issue_id,
            excerpt=f"Assignee: {assignee}",
        )
        if is_subject_allowed(c["predicate"], c["subject"]) and validate_claim(c):
            claims.append(c)
        else:
            print(f"Skipped invalid or disallowed claim: {c.get('claim_id')}")

    # Process comments: create ISSUE_HAS_COMMENT and COMMENT_WRITTEN_BY
    for comment in (issue.get("comments") or []):
        try:
            comment_id = comment.get("comment_id") or comment.get("id")
        except Exception:
            comment_id = None
        if not comment_id:
            continue
        comment_node = f"comment_{comment_id}"
        comment_event_time = comment.get("created_at") or event_time
        excerpt = (comment.get("body") or "")[:EXCERPT_LIMIT]

        # ISSUE_HAS_COMMENT
        c1 = make_claim(
            subject=issue_node,
            predicate="ISSUE_HAS_COMMENT",
            obj=comment_node,
            event_time=comment_event_time,
            source_type="issue_comment",
            source_id=comment_id,
            excerpt=excerpt,
        )
        if is_subject_allowed(c1["predicate"], c1["subject"]) and validate_claim(c1):
            claims.append(c1)
        else:
            print(f"Skipped invalid or disallowed claim: {c1.get('claim_id')}")

        # COMMENT_WRITTEN_BY
        comment_author = comment.get("author")
        if comment_author:
            obj_ns = namespace_object("COMMENT_WRITTEN_BY", comment_author)
            c2 = make_claim(
                subject=comment_node,
                predicate="COMMENT_WRITTEN_BY",
                obj=obj_ns,
                event_time=comment_event_time,
                source_type="issue_comment",
                source_id=comment_id,
                excerpt=excerpt,
            )
            if is_subject_allowed(c2["predicate"], c2["subject"]) and validate_claim(c2):
                claims.append(c2)
            else:
                print(f"Skipped invalid or disallowed claim: {c2.get('claim_id')}")

        # Check comment normalized body for assignment patterns like 'assigned @username'
        normalized = comment.get("normalized_body") or comment.get("body") or ""
        try:
            if re.search(r"\bassign\w*\b", normalized, flags=re.IGNORECASE):
                mentions = re.findall(r"@([A-Za-z0-9_\-]+)", normalized)
                for m in mentions:
                    obj_ns = namespace_object("ISSUE_ASSIGNED_TO", m)
                    c3 = make_claim(
                        subject=issue_node,
                        predicate="ISSUE_ASSIGNED_TO",
                        obj=obj_ns,
                        event_time=comment_event_time,
                        source_type="issue_comment",
                        source_id=comment_id,
                        excerpt=excerpt,
                    )
                    if is_subject_allowed(c3["predicate"], c3["subject"]) and validate_claim(c3):
                        claims.append(c3)
                    else:
                        print(f"Skipped invalid or disallowed claim: {c3.get('claim_id')}")
        except Exception:
            pass

    # post-process temporal validity for status claims:
    # for each (subject, predicate==ISSUE_HAS_STATUS) sort by event_time
    # and set previous.claim.valid_to = next.event_time
    def _adjust_status_validity(all_claims: List[Dict[str, Any]]) -> None:
        buckets: Dict[str, List[int]] = {}
        for idx, c in enumerate(all_claims):
            if c.get("predicate") == "ISSUE_HAS_STATUS":
                key = c.get("subject")
                buckets.setdefault(key, []).append(idx)

        for key, indices in buckets.items():
            # sort indices by event_time ascending
            indices.sort(key=lambda i: all_claims[i].get("event_time") or "")
            for a, b in zip(indices, indices[1:]):
                next_time = all_claims[b].get("event_time")
                if next_time:
                    all_claims[a]["valid_to"] = next_time

    _adjust_status_validity(claims)

    os.makedirs(os.path.dirname(OUTPUT_FILE) or "data/claims", exist_ok=True)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(claims, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write claims to {OUTPUT_FILE}: {e}")
        return

    print(f"Generated {len(claims)} claims.")


if __name__ == "__main__":
    main()
