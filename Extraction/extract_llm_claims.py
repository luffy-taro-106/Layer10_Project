"""Extract ISSUE_ASSIGNED_TO claims using an LLM (Ollama).

This script reads processed issues/comments and asks an LLM for assignment
claims in a strict JSON format. Output claims follow the same ontology and
validation rules used by claims.py.
"""

import json
import uuid
import os
import re
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import argparse

from ollama import chat #type: ignore
from claims import namespace_object, validate_claim, is_subject_allowed
try:
    from tqdm import tqdm ##type: ignore
    HAVE_TQDM = True
except Exception:
    def tqdm(x, **kwargs):
        return x
    HAVE_TQDM = False

INPUT_FILE = "/Users/luffy_sama/Desktop/Workspace/Try/10/data/processed/airflow_clean.json"
OUTPUT_FILE = "/Users/luffy_sama/Desktop/Workspace/Try/10/data/claims/claims_llm_v1.json"
MODEL = "llama3"
EXCERPT_LIMIT = 300
SCHEMA_PATH = "/Users/luffy_sama/Desktop/Workspace/Try/10/data/claims/schema_v1.json"
MAX_RETRIES = 2
MIN_CONFIDENCE = 0.6
TRIAL_ISSUE_LIMIT = 200
INVALID_ASSIGNEE_TOKENS = {
    "",
    "none",
    "null",
    "na",
    "n_a",
    "n/a",
    "unknown",
    "username",
    "user",
    "user_username",
    "your_username",
    "your-username",
    "assignee",
}


def _stable_claim_id(subject: str, predicate: str, obj: str, event_time: str, source_id: Any) -> str:
    # Keep claim IDs aligned with claims.py for consistent dedup behavior.
    key = "|".join([str(subject), str(predicate), str(obj), str(event_time), str(source_id)])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _find_offsets(source_text: Any, excerpt: str):
    if not source_text or not excerpt:
        return {"char_start": None, "char_end": None}
    try:
        s = str(source_text)
        ex = str(excerpt)
        idx = s.find(ex)
        if idx != -1:
            return {"char_start": idx, "char_end": idx + len(ex)}
        # normalized whitespace
        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x).strip()
        ns = norm(s)
        ne = norm(ex)
        idx = ns.find(ne)
        if idx != -1:
            token = ne.split()[0]
            approx_idx = s.find(token)
            if approx_idx != -1:
                return {"char_start": approx_idx, "char_end": approx_idx + len(ne)}
    except Exception:
        pass
    return {"char_start": None, "char_end": None}


def _extract_assignee_username(raw_obj: Any, evidence: Any = None) -> Optional[str]:
    # Accept only explicit @username mentions.
    sources = []
    if raw_obj is not None:
        sources.append(str(raw_obj))
    if evidence is not None:
        sources.append(str(evidence))

    candidate: Optional[str] = None
    for src in sources:
        mentions = re.findall(r"@([A-Za-z0-9][A-Za-z0-9_-]*)", src)
        if mentions:
            candidate = mentions[-1]
            break

    if not candidate:
        return None

    token = candidate.strip().lower().strip(" \t\n\r.,;:!?()[]{}<>\"'")
    if not token or token in INVALID_ASSIGNEE_TOKENS:
        return None
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", token):
        return None
    return token


def make_claim(subject, predicate, obj, event_time, source_type, source_id, excerpt, source_text=None, model=None, prompt_version=None, confidence: float = 0.85):
    excerpt_text = (excerpt or "")[:EXCERPT_LIMIT]
    claim_id = _stable_claim_id(subject, predicate, obj, event_time, source_id)
    offsets = _find_offsets(source_text, excerpt_text)
    extraction_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "claim_id": claim_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "event_time": event_time,
        "valid_from": event_time,
        "valid_to": None,
        "confidence": float(confidence),
        "extraction_metadata": {
            "schema_version": "1.0",
            "extractor_version": "llm_v1",
            "model": model,
            "prompt_version": prompt_version,
            "extracted_at": extraction_ts,
        },
        "evidence": {
            "source_type": source_type,
            "source_id": source_id,
            "excerpt": excerpt_text,
            "location": None,
            "offsets": offsets,
            "timestamp": event_time,
            "extraction": {"method": source_type, "timestamp": extraction_ts},
        },
    }


def _find_json_in_text(text: str) -> Optional[Any]:
    if not text:
        return None
    # Try to find fenced code blocks with JSON first (object or array)
    m = re.search(r"```(?:json\s*)?([\s\S]*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try to find any JSON array in the text (non-greedy)
    for match in re.finditer(r"\[.*?\]", text, flags=re.DOTALL):
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # Try to find any JSON object in the text (non-greedy)
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            continue

    return None


def call_llm(text: str) -> Optional[Dict[str, Any]]:
    # Build the prompt safely: serialize the example schema so braces aren't interpreted
    schema_example = {
        "claims": [
            {
                "predicate": "...",
                "object": "...",
                "evidence": "...",
                "confidence": "0-1"
            }
        ]
    }
    schema_text = json.dumps(schema_example, indent=2)

    prompt = (
        "You are extracting assignment claims from GitHub issue text.\n\n"
        "Return ONLY valid JSON.\n\n"
        "Schema:\n\n"
        f"{schema_text}\n\n"
        "Rules:\n"
        "- evidence must be an exact sentence from the text\n"
        "- only output predicate ISSUE_ASSIGNED_TO\n"
        "- output claims ONLY when text has explicit @username mention\n"
        "- object must be exactly that mentioned username (without @)\n"
        "- do not output placeholders like 'none' or 'your_username'\n"
        "- do not invent information\n\n"
        "Allowed predicates:\n"
        "ISSUE_ASSIGNED_TO\n\n"
        "Text:\n"
        f"{text}"
    )
    # Avoid f-string interpolation of braces above; inject the text safely
    prompt = prompt.replace("{text}", text)

    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            response = chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        except Exception:
            attempt += 1
            continue

        # Extract content from various response shapes (dict, string, or ChatResponse-like)
        content = None
        if isinstance(response, dict):
            msg = response.get("message")
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                    content = choice["message"].get("content")
                elif isinstance(choice, dict) and "text" in choice:
                    content = choice.get("text")
        elif isinstance(response, str):
            content = response
        else:
            # ChatResponse-like object: try .message.content or .content
            try:
                msg = getattr(response, "message", None)
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                elif hasattr(msg, "content"):
                    content = msg.content
                elif hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)
            except Exception:
                content = str(response)

        if not content:
            attempt += 1
            continue

        parsed = _find_json_in_text(content)

        if parsed is not None:
            if isinstance(parsed, dict) and isinstance(parsed.get("claims"), list):
                return parsed

        # As a last resort, try to parse the whole content
        try:
            parsed_whole = json.loads(content)
            if isinstance(parsed_whole, dict) and isinstance(parsed_whole.get("claims"), list):
                return parsed_whole
        except Exception:
            attempt += 1
            continue

        attempt += 1

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LLM claims (optionally test a single issue)")
    parser.add_argument("--issue-index", type=int, help="Zero-based index of issue to process for testing")
    parser.add_argument("--issue-id", help="Issue id or number to process for testing")
    args = parser.parse_args()

    if not os.path.exists(INPUT_FILE):
        return

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            issues = json.load(f)
    except Exception:
        return

    if not isinstance(issues, list):
        return

    # Support testing a single issue via CLI flags
    if args.issue_index is not None:
        if args.issue_index < 0 or args.issue_index >= len(issues):
            return
        issues = [issues[args.issue_index]]
    elif args.issue_id is not None:
        found = None
        for it in issues:
            if str(it.get("issue_id") or it.get("number")) == str(args.issue_id):
                found = it
                break
        if not found:
            return
        issues = [found]
    else:
        issues = issues[:TRIAL_ISSUE_LIMIT]

    claims = []
    total_issues = len(issues)

    for issue in tqdm(issues, desc="Issues", unit="issue"):
        issue_id = issue.get("issue_id") or issue.get("number")
        if not issue_id:
            continue
        issue_node = f"issue_{issue_id}"
        event_time = issue.get("created_at") or issue.get("updated_at") or ""

        texts = []
        body = issue.get("body") or ""
        if body:
            texts.append({
                "text": body,
                "source_type": "issue_text",
                "source_id": issue_id,
                "event_time": event_time,
            })
        for c in (issue.get("comments") or []):
            if isinstance(c, dict):
                comment_body = c.get("body") or ""
                if not comment_body:
                    continue
                texts.append({
                    "text": comment_body,
                    "source_type": "issue_comment",
                    "source_id": c.get("comment_id") or c.get("id") or issue_id,
                    "event_time": c.get("created_at") or event_time,
                })

        for src in texts:
            text = src["text"]
            if not text or len(text) < 20:
                continue

            result = call_llm(text)
            if not result:
                continue

            llm_claims = result.get("claims") if isinstance(result, dict) else None
            if not isinstance(llm_claims, list):
                continue

            for item in llm_claims:
                try:
                    if not isinstance(item, dict):
                        continue
                    predicate = item.get("predicate")
                    if predicate != "ISSUE_ASSIGNED_TO":
                        continue

                    raw_obj = item.get("object")
                    evidence = item.get("evidence")
                    confidence = item.get("confidence", MIN_CONFIDENCE)
                    if not raw_obj or not evidence:
                        continue

                    assignee_username = _extract_assignee_username(raw_obj, evidence)
                    if not assignee_username:
                        continue
                    obj_val = namespace_object("ISSUE_ASSIGNED_TO", assignee_username)
                    if obj_val in {"person_none", "person_your_username", "person_username"}:
                        continue

                    claim = make_claim(
                        subject=issue_node,
                        predicate=predicate,
                        obj=obj_val,
                        event_time=src.get("event_time") or event_time,
                        source_type=src.get("source_type") or "issue_text",
                        source_id=src.get("source_id") or issue_id,
                        excerpt=evidence,
                        source_text=text,
                        model=MODEL,
                        prompt_version="llm_structured_v1",
                        confidence=float(confidence) if confidence is not None else MIN_CONFIDENCE,
                    )

                    if not is_subject_allowed(claim["predicate"], claim["subject"]):
                        continue
                    if not validate_claim(claim, schema_path=SCHEMA_PATH):
                        continue

                    claims.append(claim)
                except Exception:
                    continue

    os.makedirs(os.path.dirname(OUTPUT_FILE) or "data/claims", exist_ok=True)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(claims, f, indent=2, ensure_ascii=False)
    except Exception:
        return


if __name__ == "__main__":
    main()
