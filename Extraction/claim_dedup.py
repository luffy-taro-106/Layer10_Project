#!/usr/bin/env python3
"""Deduplicate and merge claims according to semantic key rules.

- Deterministic claims and LLM claims are merged by (subject,predicate,object).
- Temporal predicates (ISSUE_HAS_STATUS) are only merged when valid_from matches.
- Evidence arrays are merged and support_count tracked.
- Confidence is updated based on sources and support_count.

Input: data/claims/claims_v1.json (deterministic claims) and
       data/claims/claims_llm_v1.json (LLM claims) if present.
Output: data/claims/claims_merged_v1.json
"""

import json
import os
from typing import Dict, Any, Tuple, List

INPUT_FILES = [
    "data/claims/claims_v1.json",
    "data/claims/claims_llm_v1.json",
]
OUTPUT_FILE = "data/claims/claims_dedup_v1.json"


def _load_claims(paths: List[str]) -> List[Dict[str, Any]]:
    claims = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                c = json.load(f)
                if isinstance(c, list):
                    claims.extend(c)
        except Exception:
            continue
    return claims


def _is_temporal(predicate: str) -> bool:
    return predicate == "ISSUE_HAS_STATUS"


def _source_confidence(claim: Dict[str, Any]) -> float:
    # deterministic claims have higher default confidence
    em = claim.get("extraction_metadata") or {}
    extractor = em.get("extractor_version")
    if extractor and extractor.startswith("llm"):
        return float(claim.get("confidence", 0.6))
    return float(claim.get("confidence", 1.0))


def dedup_and_merge(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    for claim in claims:
        subj = claim.get("subject")
        pred = claim.get("predicate")
        obj = claim.get("object")
        # temporal key includes valid_from for temporal predicates
        valid_from = claim.get("valid_from")
        if _is_temporal(pred):
            key = (subj, pred, obj, str(valid_from))
        else:
            key = (subj, pred, obj, "")

        existing = buckets.get(key)
        if not existing:
            # initialize merged claim
            merged = dict(claim)
            merged["supporting_evidence"] = [claim.get("evidence")]
            merged["support_count"] = 1
            # normalize confidence
            merged["confidence"] = _source_confidence(claim)
            buckets[key] = merged
        else:
            # merge evidence arrays
            existing.setdefault("supporting_evidence", [])
            existing["supporting_evidence"].append(claim.get("evidence"))
            existing["support_count"] = existing.get("support_count", 1) + 1
            # update confidence: simple heuristic
            existing_conf = existing.get("confidence", 0.0)
            new_conf = _source_confidence(claim)
            # combine confidences conservatively: 1 - (1-a)*(1-b)
            combined = 1 - (1 - existing_conf) * (1 - new_conf)
            existing["confidence"] = round(combined, 4)
            # preserve earliest event_time as event_time
            if claim.get("event_time") and (not existing.get("event_time") or claim.get("event_time") < existing.get("event_time")):
                existing["event_time"] = claim.get("event_time")

    return list(buckets.values())


def main():
    claims = _load_claims(INPUT_FILES)
    merged = dedup_and_merge(claims)

    os.makedirs(os.path.dirname(OUTPUT_FILE) or "data/claims", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(claims)} -> {len(merged)} claims. Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
