from neo4j import GraphDatabase
import json
import os
import hashlib
from typing import Any, Dict
from dotenv import load_dotenv
load_dotenv()

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    ),
)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

GITHUB_OWNER = os.getenv("GITHUB_OWNER", "apache")
GITHUB_REPO = os.getenv("GITHUB_REPO", "airflow")
CLAIMS_CANDIDATES = [
    "data/claims/claims_dedup_v1.json",
]


def _claim_key(claim: dict) -> str:
    claim_id = claim.get("claim_id")
    if claim_id:
        return f"id:{claim_id}"
    return "|".join(
        [
            str(claim.get("subject")),
            str(claim.get("predicate")),
            str(claim.get("object")),
            str(claim.get("event_time")),
            str((claim.get("evidence") or {}).get("source_id") if isinstance(claim.get("evidence"), dict) else None),
        ]
    )


def _load_claims() -> list:
    merged: dict[str, dict] = {}
    for path in CLAIMS_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                claims = json.load(f)
        except Exception:
            continue
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            key = _claim_key(claim)
            if key not in merged:
                merged[key] = claim
    return list(merged.values())


def _evidence_id(evidence: dict) -> str:
    # deterministic evidence id from source_type+source_id+excerpt
    key = f"{evidence.get('source_type')}|{evidence.get('source_id')}|{(evidence.get('excerpt') or '')[:200]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _extract_issue_number(node_id):
    if not isinstance(node_id, str):
        return None
    if node_id.startswith("issue_"):
        value = node_id.split("issue_", 1)[1].strip()
        return value or None
    return None


def _extract_comment_id(node_id):
    if not isinstance(node_id, str):
        return None
    if node_id.startswith("comment_"):
        value = node_id.split("comment_", 1)[1].strip()
        return value or None
    return None


def _issue_url(owner: str, repo: str, issue_number: str) -> str:
    return f"https://github.com/{owner}/{repo}/issues/{issue_number}"


def _comment_url(owner: str, repo: str, issue_number: str, comment_id: str) -> str:
    return f"https://github.com/{owner}/{repo}/issues/{issue_number}#issuecomment-{comment_id}"


def _build_comment_issue_map(claims):
    comment_to_issue = {}
    for claim in claims:
        if claim.get("predicate") != "ISSUE_HAS_COMMENT":
            continue
        issue_number = _extract_issue_number(claim.get("subject"))
        comment_id = _extract_comment_id(claim.get("object"))
        if issue_number and comment_id:
            comment_to_issue[str(comment_id)] = str(issue_number)
    return comment_to_issue


def _evidence_url(claim: dict, owner: str, repo: str, comment_issue_map: dict) -> str:
    ev = claim.get("evidence") or {}
    source_type = ev.get("source_type")
    source_id = ev.get("source_id")

    if source_type in ("issue_text", "issue_metadata"):
        issue_number = str(source_id) if source_id is not None else _extract_issue_number(claim.get("subject"))
        if issue_number:
            return _issue_url(owner, repo, issue_number)

    if source_type == "issue_comment":
        comment_id = str(source_id) if source_id is not None else _extract_comment_id(claim.get("subject")) or _extract_comment_id(claim.get("object"))
        issue_number = comment_issue_map.get(str(comment_id)) if comment_id else None
        if not issue_number:
            issue_number = _extract_issue_number(claim.get("subject"))
        if issue_number and comment_id:
            return _comment_url(owner, repo, issue_number, comment_id)
        if issue_number:
            return _issue_url(owner, repo, issue_number)

    fallback_issue = _extract_issue_number(claim.get("subject"))
    if fallback_issue:
        return _issue_url(owner, repo, fallback_issue)
    return f"https://github.com/{owner}/{repo}"


def create_evidence(tx, claim: dict, owner: str, repo: str, comment_issue_map: dict) -> Dict[str, Any]:
    ev = claim.get("evidence") or {}
    if not ev:
        return {"has_evidence": False, "claim_found": False, "linked": False}
    eid = _evidence_id(ev)
    # flatten nested structures into primitive properties (Neo4j disallows maps as property values)
    ev_props: dict = {}
    for k, v in ev.items():
        if isinstance(v, (str, int, float)) or v is None or isinstance(v, bool):
            ev_props[k] = v
        elif isinstance(v, dict):
            # handle known nested objects specially
            if k == "offsets":
                ev_props["offsets_char_start"] = v.get("char_start")
                ev_props["offsets_char_end"] = v.get("char_end")
            elif k == "extraction":
                ev_props["extraction_method"] = v.get("method")
                ev_props["extraction_timestamp"] = v.get("timestamp")
            else:
                # fallback: serialize nested object to JSON string
                try:
                    ev_props[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    ev_props[k] = str(v)
        else:
            # fallback for lists or unknown types
            try:
                ev_props[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                ev_props[k] = str(v)

    ev_props["evidence_id"] = eid
    ev_props["url"] = _evidence_url(claim, owner, repo, comment_issue_map)
    # create/merge evidence node (properties are now primitives/arrays)
    tx.run("MERGE (e:Evidence {evidence_id:$eid}) SET e += $props", eid=eid, props=ev_props)
    # link claim -> evidence
    cid = claim.get("claim_id")
    rec = tx.run(
        """
        OPTIONAL MATCH (c:Claim {claim_id:$cid})
        MATCH (e:Evidence {evidence_id:$eid})
        FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END |
            MERGE (c)-[:SUPPORTED_BY]->(e)
        )
        RETURN c IS NOT NULL AS claim_found
        """,
        cid=cid,
        eid=eid,
    ).single()
    claim_found = bool(rec["claim_found"]) if rec else False
    return {"has_evidence": True, "claim_found": claim_found, "linked": claim_found}


def main():
    claims = _load_claims()
    if not claims:
        print("Claims file not found.")
        return

    comment_issue_map = _build_comment_issue_map(claims)

    linked = 0
    missing_claim = 0
    no_evidence = 0

    with driver.session(database=NEO4J_DATABASE) as session:
        for c in tqdm(claims, desc="Evidence", total=len(claims)):
            result = session.execute_write(create_evidence, c, GITHUB_OWNER, GITHUB_REPO, comment_issue_map)
            if not result:
                continue
            if not result.get("has_evidence"):
                no_evidence += 1
                continue
            if result.get("linked"):
                linked += 1
            elif not result.get("claim_found"):
                missing_claim += 1

    print(f"Loaded evidence for {len(claims)} claims into Neo4j.")
    print(f"Evidence links created: {linked}")
    print(f"Claims missing during link: {missing_claim}")
    print(f"Claims without evidence payload: {no_evidence}")


if __name__ == "__main__":
    main()
