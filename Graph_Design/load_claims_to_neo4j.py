from neo4j import GraphDatabase
import json
import os
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

GITHUB_OWNER = os.getenv("GITHUB_OWNER", "apache")
GITHUB_REPO = os.getenv("GITHUB_REPO", "airflow")
CLAIMS_CANDIDATES = [
    "data/claims/claims_dedup_v1.json",
]


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


def _claim_key(claim: Dict[str, Any]) -> str:
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
    merged: Dict[str, Dict[str, Any]] = {}
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


def _claim_url(claim: dict, owner: str, repo: str, comment_issue_map: dict) -> str:
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


def _infer_entity_type(node_id: Any) -> str:
    if not isinstance(node_id, str):
        return "Unknown"
    if node_id.startswith("issue_"):
        return "Issue"
    if node_id.startswith("comment_"):
        return "Comment"
    if node_id.startswith("person_"):
        return "Person"
    if node_id.startswith("label_"):
        return "Label"
    if node_id.startswith("status_"):
        return "Status"
    if node_id.startswith("pr_"):
        return "PullRequest"
    return "Unknown"


def create_claim_and_edge(tx, claim: dict, owner: str, repo: str, comment_issue_map: dict):
    ev = claim.get("evidence") or {}
    # create claim node
    claim_props = {
        "claim_id": claim.get("claim_id"),
        "predicate": claim.get("predicate"),
        "confidence": float(claim.get("confidence", 0.0)),
        "valid_from": claim.get("valid_from"),
        "valid_to": claim.get("valid_to"),
        "event_time": claim.get("event_time"),
        "url": _claim_url(claim, owner, repo, comment_issue_map),
        # Keep minimal inline evidence on claim for grounded fallback retrieval.
        "source_type": ev.get("source_type"),
        "source_id": ev.get("source_id"),
        "evidence_excerpt": ev.get("excerpt"),
        "evidence_timestamp": ev.get("timestamp"),
        "source_url": _claim_url(claim, owner, repo, comment_issue_map),
    }
    tx.run("MERGE (c:Claim {claim_id: $claim_id}) SET c += $props", claim_id=claim_props["claim_id"], props=claim_props)

    subj = claim.get("subject")
    obj = claim.get("object")

    # link subject -> claim and claim -> object
    if subj:
        stype = _infer_entity_type(subj)
        tx.run(
            """
            MERGE (s {id:$sid})
            ON CREATE SET s.entity_type = $stype, s.entity_version = '1.0', s.created_at = $event_time
            SET s.last_seen_at = coalesce($event_time, s.last_seen_at)
            """,
            sid=subj,
            stype=stype,
            event_time=claim_props.get("event_time"),
        )
        tx.run("MATCH (s {id:$sid}), (c:Claim {claim_id:$cid}) MERGE (s)-[:SUBJECT_OF]->(c)", sid=subj, cid=claim_props["claim_id"])
    if obj:
        otype = _infer_entity_type(obj)
        tx.run(
            """
            MERGE (o {id:$oid})
            ON CREATE SET o.entity_type = $otype, o.entity_version = '1.0', o.created_at = $event_time
            SET o.last_seen_at = coalesce($event_time, o.last_seen_at)
            """,
            oid=obj,
            otype=otype,
            event_time=claim_props.get("event_time"),
        )
        tx.run("MATCH (c:Claim {claim_id:$cid}), (o {id:$oid}) MERGE (c)-[:OBJECT_OF]->(o)", cid=claim_props["claim_id"], oid=obj)

    # predicate relationships are represented by Claim nodes; do not create direct subject->object relations


def main():
    claims = _load_claims()
    if not claims:
        print("Claims file not found.")
        return
    comment_issue_map = _build_comment_issue_map(claims)

    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            for c in tqdm(claims, desc="Claims", total=len(claims)):
                session.execute_write(create_claim_and_edge, c, GITHUB_OWNER, GITHUB_REPO, comment_issue_map)
    except Exception as e:
        print("Write error:", e)

    print(f"Loaded {len(claims)} claims into Neo4j.")


if __name__ == "__main__":
    main()
