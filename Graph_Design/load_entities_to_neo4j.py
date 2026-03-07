from neo4j import GraphDatabase
import json
import os
import re
from urllib.parse import quote, quote_plus
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


def _safe_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


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


def _user_url(username: str) -> str:
    return f"https://github.com/{quote(username.strip('@'), safe='')}"


def _label_name_from_entity(entity_id: str) -> str:
    token = entity_id.replace("label_", "", 1)
    prefixes = ("kind_", "area_", "priority_", "component_")
    for prefix in prefixes:
        if token.startswith(prefix):
            rest = token[len(prefix):].replace("_", "-")
            return f"{prefix[:-1]}:{rest}" if rest else prefix[:-1]
    return token.replace("_", "-")


def _build_comment_issue_map() -> dict:
    infile = next((p for p in CLAIMS_CANDIDATES if os.path.exists(p)), None)
    if not infile:
        return {}
    try:
        with open(infile, "r", encoding="utf-8") as f:
            claims = json.load(f)
    except Exception:
        return {}
    if not isinstance(claims, list):
        return {}

    comment_to_issue = {}
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if claim.get("predicate") != "ISSUE_HAS_COMMENT":
            continue
        issue_number = _extract_issue_number(claim.get("subject"))
        comment_id = _extract_comment_id(claim.get("object"))
        if issue_number and comment_id:
            comment_to_issue[str(comment_id)] = str(issue_number)
    return comment_to_issue


def _entity_url(entity: dict, owner: str, repo: str, comment_issue_map: dict) -> str:
    entity_id = entity.get("entity_id")
    if not isinstance(entity_id, str):
        return f"https://github.com/{owner}/{repo}"

    issue_number = _extract_issue_number(entity_id)
    if issue_number:
        return _issue_url(owner, repo, issue_number)

    comment_id = _extract_comment_id(entity_id)
    if comment_id:
        issue_for_comment = comment_issue_map.get(str(comment_id))
        if issue_for_comment:
            return _comment_url(owner, repo, issue_for_comment, str(comment_id))
        return f"https://github.com/{owner}/{repo}/issues"

    if entity_id.startswith("person_"):
        person_token = entity_id.split("person_", 1)[1]
        username = person_token.split("_@", 1)[1] if "_@" in person_token else person_token
        username = username.strip().strip("@")
        if username:
            return _user_url(username)
        return f"https://github.com/{owner}/{repo}"

    if entity_id.startswith("label_"):
        label_name = _label_name_from_entity(entity_id)
        return f"https://github.com/{owner}/{repo}/labels/{quote(label_name, safe=':')}"

    if entity_id.startswith("status_"):
        state = entity_id.split("status_", 1)[1] or "open"
        query = quote_plus(f"is:issue is:{state}")
        return f"https://github.com/{owner}/{repo}/issues?q={query}"

    return f"https://github.com/{owner}/{repo}"


def create_entity(tx, entity: dict, owner: str, repo: str, comment_issue_map: dict):
    label = _safe_label(entity.get("entity_type", "Entity"))
    props = dict(entity)
    props["url"] = _entity_url(entity, owner, repo, comment_issue_map)
    props_id = props.pop("entity_id")
    # Merge node by id and set/merge properties
    query = f"MERGE (e:{label} {{id: $id}}) SET e += $props"
    tx.run(query, id=props_id, props=props)


def main():
    # prefer graph path, fall back to Entities
    candidates = ["data/graph/entities_v1.json", "data/Entities/entities_v1.json", "data/Entities/entities_v1.json"]
    infile = next((p for p in candidates if os.path.exists(p)), None)
    if not infile:
        print("No entities file found; expected data/graph/entities_v1.json or data/Entities/entities_v1.json")
        return

    with open(infile, "r", encoding="utf-8") as f:
        entities = json.load(f)
    comment_issue_map = _build_comment_issue_map()

    with driver.session() as session:
        for ent in tqdm(entities, desc="Entities", total=len(entities)):
            session.execute_write(create_entity, ent, GITHUB_OWNER, GITHUB_REPO, comment_issue_map)

    print(f"Loaded {len(entities)} entities into Neo4j.")


if __name__ == "__main__":
    main()
