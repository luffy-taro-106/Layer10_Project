#!/usr/bin/env python3
"""Retrieve grounded context packs from Neo4j memory graph.

Implements two retrieval strategies:
1) keyword: question -> entity/predicate hints via regex + keywords
2) hybrid: keyword candidate generation + Ollama embedding re-rank

Returns a grounded context pack with:
- ranked entities and claims
- evidence snippets with URLs/citations
- ambiguity and conflict handling
- optional Ollama-generated answer constrained to citations
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv #type: ignore
from neo4j import GraphDatabase # type: ignore
from ollama import chat  # type: ignore
import ollama  # type: ignore

load_dotenv()


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "with",
    "by",
    "who",
    "what",
    "when",
    "where",
    "how",
    "why",
    "issue",
    "comment",
}

PREDICATE_HINT_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("ISSUE_CREATED_BY", ("create", "created", "opened", "reported by")),
    ("ISSUE_ASSIGNED_TO", ("assign", "assigned", "owner", "responsible")),
    ("ISSUE_HAS_STATUS", ("status", "state", "open", "closed", "resolved")),
    ("ISSUE_HAS_LABEL", ("label", "labels", "tag", "kind:", "area:", "priority:")),
    ("ISSUE_HAS_COMMENT", ("comment", "comments", "discussion")),
    ("COMMENT_WRITTEN_BY", ("wrote", "writer", "author of comment")),
]


def _parse_iso_utc(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _extract_chat_content(resp: Any) -> str:
    if isinstance(resp, dict):
        msg = resp.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        if isinstance(resp.get("response"), str):
            return resp["response"]

    # ollama ChatResponse-like object
    msg_obj = getattr(resp, "message", None)
    if msg_obj is not None:
        content = getattr(msg_obj, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(msg_obj, dict) and isinstance(msg_obj.get("content"), str):
            return msg_obj["content"]

    direct = getattr(resp, "content", None)
    if isinstance(direct, str):
        return direct

    return str(resp)


class RetrievalGroundingService:
    def __init__(
        self,
        mode: str,
        llm_model: str,
        embedding_model: str,
        max_claims_per_entity: int,
        max_evidence_per_claim: int,
        max_claims_total: int,
        candidate_entity_limit: int,
    ) -> None:
        self.mode = mode
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.max_claims_per_entity = max_claims_per_entity
        self.max_evidence_per_claim = max_evidence_per_claim
        self.max_claims_total = max_claims_total
        self.candidate_entity_limit = candidate_entity_limit

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        self.database = os.getenv("NEO4J_DATABASE")

    def close(self) -> None:
        self.driver.close()

    def _session(self):
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def _detect_question_signals(self, question: str) -> Dict[str, Any]:
        ql = question.lower()

        issue_ids = sorted(set(re.findall(r"\bissue\s*#?\s*(\d+)\b", ql)))
        hash_issue_ids = sorted(set(re.findall(r"(?<!\w)#(\d+)\b", ql)))
        issue_ids = sorted(set(issue_ids + hash_issue_ids))

        comment_ids = sorted(set(re.findall(r"\bcomment\s*#?\s*(\d+)\b", ql)))
        usernames = sorted(set(re.findall(r"@([a-zA-Z0-9][a-zA-Z0-9_-]*)", question)))

        labels = []
        for m in re.findall(r"\b(kind|area|priority|component)\s*:\s*([a-zA-Z0-9_-]+)", ql):
            labels.append(f"label_{m[0]}_{m[1]}")

        predicate_hints: List[str] = []
        for predicate, keywords in PREDICATE_HINT_RULES:
            if any(k in ql for k in keywords):
                predicate_hints.append(predicate)

        strict_predicates: List[str] = []
        if re.search(r"\bwho\s+(created|opened|reported)\b", ql):
            strict_predicates = ["ISSUE_CREATED_BY"]
        elif re.search(r"\bwho\s+(is\s+)?assigned\b", ql) or re.search(r"\bwho\s+owns?\b", ql):
            strict_predicates = ["ISSUE_ASSIGNED_TO"]

        explicit_entities = [f"issue_{i}" for i in issue_ids]
        explicit_entities.extend(f"comment_{c}" for c in comment_ids)
        explicit_entities.extend(f"person_{u.lower()}" for u in usernames)
        explicit_entities.extend(labels)

        query_terms = [
            t
            for t in re.findall(r"[a-zA-Z0-9_:-]+", ql)
            if len(t) >= 3 and t not in STOPWORDS
        ]

        return {
            "issue_ids": issue_ids,
            "comment_ids": comment_ids,
            "usernames": usernames,
            "explicit_entities": explicit_entities,
            "predicate_hints": sorted(set(predicate_hints)),
            "strict_predicates": strict_predicates,
            "query_terms": sorted(set(query_terms)),
        }

    @staticmethod
    def _infer_entity_type(node_id: str) -> str:
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

    def _fetch_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (e {id:$entity_id})
        RETURN e.id AS id, coalesce(e.entity_type, 'Unknown') AS entity_type, coalesce(e.url, '') AS url
        LIMIT 1
        """
        with self._session() as session:
            rec = session.run(query, entity_id=entity_id).single()
            if rec:
                return {"id": rec["id"], "entity_type": rec["entity_type"], "url": rec["url"], "score": 3.0}
        # entity might not exist yet; still keep explicit signal
        return {"id": entity_id, "entity_type": self._infer_entity_type(entity_id), "url": "", "score": 2.0}

    def _search_entities_keyword(self, terms: List[str], limit: int) -> List[Dict[str, Any]]:
        if not terms:
            return []
        query = """
        MATCH (e)
        WHERE e.id IS NOT NULL
          AND any(t IN $terms WHERE toLower(e.id) CONTAINS t)
        RETURN e.id AS id,
               coalesce(e.entity_type, 'Unknown') AS entity_type,
               coalesce(e.url, '') AS url,
               reduce(score = 0.0, t IN $terms |
                 score + CASE WHEN toLower(e.id) CONTAINS t THEN 1.0 ELSE 0.0 END
               ) AS score
        ORDER BY score DESC, id ASC
        LIMIT $limit
        """
        with self._session() as session:
            rows = session.run(query, terms=terms, limit=limit)
            return [
                {
                    "id": row["id"],
                    "entity_type": row["entity_type"],
                    "url": row["url"],
                    "score": float(row["score"] or 0.0),
                }
                for row in rows
            ]

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return None
        try:
            if hasattr(ollama, "embed"):
                resp = ollama.embed(model=self.embedding_model, input=texts)
                embeddings = resp.get("embeddings")
                if isinstance(embeddings, list) and embeddings:
                    return embeddings
            if hasattr(ollama, "embeddings"):
                out: List[List[float]] = []
                for t in texts:
                    resp = ollama.embeddings(model=self.embedding_model, prompt=t)
                    emb = resp.get("embedding")
                    if not isinstance(emb, list):
                        return None
                    out.append(emb)
                return out
        except Exception:
            return None
        return None

    def _rerank_entities_hybrid(self, question: str, entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        if not entities:
            return entities, False
        texts = [question] + [f"{e['id']} {e.get('entity_type', '')}" for e in entities]
        embeddings = self._embed_texts(texts)
        if not embeddings or len(embeddings) != len(texts):
            return entities, False

        q_emb = embeddings[0]
        ranked = []
        for ent, emb in zip(entities, embeddings[1:]):
            sim = _cosine_similarity(q_emb, emb)
            hybrid_score = float(ent.get("score", 0.0)) + (sim * 2.0)
            item = dict(ent)
            item["embedding_similarity"] = sim
            item["score"] = hybrid_score
            ranked.append(item)
        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked, True

    def _candidate_entities(self, question: str, signals: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        combined: Dict[str, Dict[str, Any]] = {}

        for entity_id in signals["explicit_entities"]:
            ent = self._fetch_entity_by_id(entity_id)
            if not ent:
                continue
            combined[ent["id"]] = ent

        keyword_entities = self._search_entities_keyword(
            signals["query_terms"], self.candidate_entity_limit
        )
        for ent in keyword_entities:
            cur = combined.get(ent["id"])
            if not cur or ent["score"] > cur.get("score", 0.0):
                combined[ent["id"]] = ent

        entities = list(combined.values())
        entities.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        hybrid_used = False
        if self.mode == "hybrid":
            entities, hybrid_used = self._rerank_entities_hybrid(question, entities)

        return entities[: self.candidate_entity_limit], {
            "hybrid_embeddings_used": hybrid_used if self.mode == "hybrid" else False
        }

    def _fetch_claims_for_entity(
        self, entity_id: str, predicate_hints: List[str]
    ) -> List[Dict[str, Any]]:
        query = """
        MATCH (s {id:$entity_id})-[:SUBJECT_OF]->(c:Claim)
        OPTIONAL MATCH (c)-[:OBJECT_OF]->(o)
        WHERE $use_predicates = false OR c.predicate IN $predicates
        RETURN c.claim_id AS claim_id,
               s.id AS subject,
               coalesce(s.entity_type, 'Unknown') AS subject_type,
               coalesce(s.url, '') AS subject_url,
               c.predicate AS predicate,
               o.id AS object,
               coalesce(o.entity_type, 'Unknown') AS object_type,
               coalesce(o.url, '') AS object_url,
               c.confidence AS confidence,
               c.event_time AS event_time,
               properties(c)['valid_from'] AS valid_from,
               properties(c)['valid_to'] AS valid_to,
               coalesce(properties(c)['url'], '') AS claim_url
        ORDER BY c.event_time DESC
        LIMIT $limit
        """
        with self._session() as session:
            rows = session.run(
                query,
                entity_id=entity_id,
                use_predicates=bool(predicate_hints),
                predicates=predicate_hints,
                limit=self.max_claims_per_entity,
            )
            return [dict(r) for r in rows]

    def _fetch_evidence_for_claim(self, claim_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (c:Claim {claim_id:$claim_id})-[:SUPPORTED_BY]->(e:Evidence)
        RETURN e.evidence_id AS evidence_id,
               e.source_type AS source_type,
               e.source_id AS source_id,
               e.excerpt AS excerpt,
               e.timestamp AS timestamp,
               coalesce(e.url, '') AS url
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        with self._session() as session:
            rows = session.run(query, claim_id=claim_id, limit=self.max_evidence_per_claim)
            evidence_rows = [dict(r) for r in rows]
            if evidence_rows:
                return evidence_rows

            inline = self._fetch_inline_evidence_for_claim(claim_id)
            if inline:
                return [inline]
            return []

    def _fetch_inline_evidence_for_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (c:Claim {claim_id:$claim_id})
        RETURN
            ('inline:' + c.claim_id) AS evidence_id,
            coalesce(c.source_type, 'claim_inline') AS source_type,
            coalesce(c.source_id, c.subject) AS source_id,
            coalesce(
                c.evidence_excerpt,
                'No separate Evidence node found. Using claim-level source reference.'
            ) AS excerpt,
            coalesce(c.evidence_timestamp, c.event_time) AS timestamp,
            coalesce(c.source_url, c.url, '') AS url
        LIMIT 1
        """
        with self._session() as session:
            rec = session.run(query, claim_id=claim_id).single()
            if not rec:
                return None
            row = dict(rec)
            if not row.get("url") and not row.get("excerpt"):
                return None
            return row

    @staticmethod
    def _evidence_strength(ev: Dict[str, Any], predicate: Optional[str]) -> float:
        score = 0.0
        source_type = str(ev.get("source_type") or "")
        excerpt = str(ev.get("excerpt") or "").strip()
        low_excerpt = excerpt.lower()

        if source_type in {"issue_metadata", "issue_text", "issue_comment", "claim_inline"}:
            score += 1.0
        if len(excerpt) >= 20:
            score += 0.4
        elif len(excerpt) >= 8:
            score += 0.1

        # generic metadata snippets are weak for natural QA
        if low_excerpt.startswith("state:") or low_excerpt.startswith("label:"):
            score -= 0.6
        if predicate == "ISSUE_CREATED_BY" and (low_excerpt.startswith("state:") or low_excerpt.startswith("label:")):
            score -= 0.6
        return score

    def _select_strong_evidence(
        self, evidences: List[Dict[str, Any]], predicate: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not evidences:
            return []
        ranked = sorted(
            evidences,
            key=lambda ev: self._evidence_strength(ev, predicate),
            reverse=True,
        )
        strong = [ev for ev in ranked if self._evidence_strength(ev, predicate) >= 0.8]
        if strong:
            return strong[: self.max_evidence_per_claim]
        return ranked[: self.max_evidence_per_claim]

    @staticmethod
    def _recency_bonus(event_time: Optional[str]) -> float:
        dt = _parse_iso_utc(event_time)
        if not dt:
            return 0.0
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
        if days <= 30:
            return 0.2
        if days <= 180:
            return 0.1
        if days <= 365:
            return 0.05
        return 0.0

    def _rank_claim(
        self,
        claim: Dict[str, Any],
        evidence_count: int,
        entity_score: float,
        predicate_hints: List[str],
    ) -> float:
        conf = float(claim.get("confidence") or 0.0)
        support_bonus = min(evidence_count, self.max_evidence_per_claim) * 0.1
        recency = self._recency_bonus(claim.get("event_time"))
        predicate_bonus = 0.2 if claim.get("predicate") in predicate_hints else 0.0
        entity_bonus = min(entity_score, 3.0) * 0.05
        return round(conf + support_bonus + recency + predicate_bonus + entity_bonus, 4)

    def _diversify_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not claims:
            return []
        claims_sorted = sorted(claims, key=lambda c: c["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        used = set()

        # first pass: keep top per predicate for diversity
        seen_predicates = set()
        for c in claims_sorted:
            pred = c.get("predicate")
            if pred in seen_predicates:
                continue
            selected.append(c)
            used.add(c["claim_id"])
            seen_predicates.add(pred)
            if len(selected) >= self.max_claims_total:
                return selected

        # second pass: fill by score
        for c in claims_sorted:
            if c["claim_id"] in used:
                continue
            selected.append(c)
            used.add(c["claim_id"])
            if len(selected) >= self.max_claims_total:
                break
        return selected

    @staticmethod
    def _detect_conflicts(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for c in claims:
            key = (str(c.get("subject")), str(c.get("predicate")))
            b = buckets.setdefault(key, {"objects": set(), "claim_ids": []})
            b["objects"].add(str(c.get("object")))
            b["claim_ids"].append(c.get("claim_id"))

        conflicts = []
        for (subject, predicate), info in buckets.items():
            objs = sorted(info["objects"])
            if len(objs) > 1:
                conflicts.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "objects": objs,
                        "claim_ids": info["claim_ids"],
                    }
                )
        return conflicts

    def _build_ambiguity(
        self, question: str, signals: Dict[str, Any], candidate_entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if signals["issue_ids"] or signals["comment_ids"]:
            return {"is_ambiguous": False}

        ql = question.lower()
        if "issue" not in ql:
            return {"is_ambiguous": False}

        issue_candidates = [e for e in candidate_entities if str(e.get("id", "")).startswith("issue_")]
        if len(issue_candidates) <= 1:
            return {"is_ambiguous": False}

        top = [e["id"] for e in issue_candidates[:5]]
        return {
            "is_ambiguous": True,
            "follow_up": "I found multiple issues. Did you mean one of: " + ", ".join(top),
            "candidate_issues": top,
        }

    def retrieve_context_pack(self, question: str) -> Dict[str, Any]:
        signals = self._detect_question_signals(question)
        candidate_entities, mode_meta = self._candidate_entities(question, signals)

        claims_by_id: Dict[str, Dict[str, Any]] = {}
        evidences_by_claim: Dict[str, List[Dict[str, Any]]] = {}
        skipped_claims_without_evidence = 0

        for ent in candidate_entities:
            entity_id = ent["id"]
            claim_rows = self._fetch_claims_for_entity(entity_id, signals["predicate_hints"])
            for claim in claim_rows:
                claim_id = claim.get("claim_id")
                if not claim_id:
                    continue
                if claim_id not in evidences_by_claim:
                    raw_evidences = self._fetch_evidence_for_claim(claim_id)
                    evidences_by_claim[claim_id] = self._select_strong_evidence(
                        raw_evidences, claim.get("predicate")
                    )
                evidences = evidences_by_claim.get(claim_id, [])
                if not evidences:
                    # enforce grounding: skip claims with no evidence
                    skipped_claims_without_evidence += 1
                    continue

                score = self._rank_claim(
                    claim,
                    evidence_count=len(evidences),
                    entity_score=float(ent.get("score", 0.0)),
                    predicate_hints=signals["predicate_hints"],
                )
                current = claims_by_id.get(claim_id)
                candidate = dict(claim)
                candidate["score"] = score
                candidate["support_count"] = len(evidences)
                if (not current) or candidate["score"] > current["score"]:
                    claims_by_id[claim_id] = candidate

        ranked_claims = list(claims_by_id.values())
        # If we have strict/explicit predicate intent, remove unrelated claim types.
        strict_preds = signals.get("strict_predicates") or []
        hint_preds = signals.get("predicate_hints") or []
        if strict_preds:
            strict_only = [c for c in ranked_claims if c.get("predicate") in strict_preds]
            if strict_only:
                ranked_claims = strict_only
        elif hint_preds:
            hinted = [c for c in ranked_claims if c.get("predicate") in hint_preds]
            if hinted:
                ranked_claims = hinted

        selected_claims = self._diversify_claims(ranked_claims)

        # build citation-first evidence list
        all_evidence: List[Dict[str, Any]] = []
        citation_index = 1
        for claim in selected_claims:
            claim_id = claim["claim_id"]
            claim_citations = []
            for ev in evidences_by_claim.get(claim_id, [])[: self.max_evidence_per_claim]:
                citation_id = f"E{citation_index}"
                citation_index += 1
                ev_item = {
                    "citation_id": citation_id,
                    "claim_id": claim_id,
                    "source_type": ev.get("source_type"),
                    "source_id": ev.get("source_id"),
                    "excerpt": ev.get("excerpt"),
                    "timestamp": ev.get("timestamp"),
                    "url": ev.get("url"),
                }
                claim_citations.append(citation_id)
                all_evidence.append(ev_item)
            claim["citations"] = claim_citations

        conflicts = self._detect_conflicts(selected_claims)
        ambiguity = self._build_ambiguity(question, signals, candidate_entities)
        diagnostics: Dict[str, Any] = {}
        if not selected_claims:
            with self._session() as session:
                inv = session.run(
                    """
                    MATCH (c:Claim) WITH count(c) AS claim_count
                    MATCH (e:Evidence) WITH claim_count, count(e) AS evidence_count
                    RETURN claim_count, evidence_count
                    """
                ).single()
            claim_count = int(inv["claim_count"]) if inv else 0
            evidence_count = int(inv["evidence_count"]) if inv else 0
            diagnostics = {
                "graph_inventory": {
                    "claim_nodes": claim_count,
                    "evidence_nodes": evidence_count,
                },
                "skipped_claims_without_evidence": skipped_claims_without_evidence,
            }
            if claim_count == 0 or evidence_count == 0:
                diagnostics["hint"] = (
                    "Neo4j seems to be missing claim/evidence ingestion. "
                    "Run Graph_Design/load_claims_to_neo4j.py and Graph_Design/load_evidence_to_neo4j.py"
                )

        context_pack = {
            "question": question,
            "retrieval_mode": self.mode,
            "mapping": {
                "detected_issue_ids": signals["issue_ids"],
                "detected_comment_ids": signals["comment_ids"],
                "detected_usernames": signals["usernames"],
                "predicate_hints": signals["predicate_hints"],
                "strict_predicates": signals["strict_predicates"],
                "explicit_entities": signals["explicit_entities"],
                "query_terms": signals["query_terms"],
            },
            "strategy": {
                "candidate_mapping": "keyword"
                if self.mode == "keyword"
                else "hybrid(keyword + ollama embeddings)",
                "max_claims_per_entity": self.max_claims_per_entity,
                "max_evidence_per_claim": self.max_evidence_per_claim,
                "max_claims_total": self.max_claims_total,
                "hybrid_embeddings_used": mode_meta.get("hybrid_embeddings_used", False),
                "grounding_rule": "claims without evidence are excluded",
            },
            "ambiguity": ambiguity,
            "conflicts": conflicts,
            "entities": candidate_entities,
            "claims": [
                {
                    "claim_id": c.get("claim_id"),
                    "subject": c.get("subject"),
                    "subject_url": c.get("subject_url"),
                    "predicate": c.get("predicate"),
                    "object": c.get("object"),
                    "object_url": c.get("object_url"),
                    "confidence": c.get("confidence"),
                    "event_time": c.get("event_time"),
                    "valid_from": c.get("valid_from"),
                    "valid_to": c.get("valid_to"),
                    "url": c.get("claim_url"),
                    "score": c.get("score"),
                    "support_count": c.get("support_count"),
                    "citations": c.get("citations", []),
                }
                for c in selected_claims
            ],
            "evidence": all_evidence,
        }
        if diagnostics:
            context_pack["diagnostics"] = diagnostics
        return context_pack

    def generate_answer(self, context_pack: Dict[str, Any]) -> Dict[str, Any]:
        question = context_pack.get("question", "")
        ambiguity = context_pack.get("ambiguity", {})
        if ambiguity.get("is_ambiguous"):
            return {
                "answer": ambiguity.get("follow_up"),
                "used_citations": [],
                "mode": "follow_up",
            }

        claims = context_pack.get("claims", [])
        evidence = context_pack.get("evidence", [])
        if not claims or not evidence:
            return {
                "answer": "I could not find grounded evidence for that question.",
                "used_citations": [],
                "mode": "insufficient_grounding",
            }

        evidence_lines = []
        for ev in evidence:
            evidence_lines.append(
                f"{ev['citation_id']}: [{ev.get('source_type')}:{ev.get('source_id')}] "
                f"{ev.get('excerpt')} (url: {ev.get('url')})"
            )

        claim_lines = []
        for c in claims:
            claim_lines.append(
                f"{c.get('claim_id')}: {c.get('subject')} -[{c.get('predicate')}]-> {c.get('object')} "
                f"(confidence={c.get('confidence')}, citations={c.get('citations')})"
            )

        prompt = (
            "You answer questions ONLY from the provided grounded context pack.\n"
            "Rules:\n"
            "- Use only facts present in claims/evidence.\n"
            "- Include inline citations like [E1], [E2] for every factual statement.\n"
            "- If conflicting claims exist, state both and do not fabricate a winner.\n"
            "- If insufficient evidence, say so.\n\n"
            "Output style:\n"
            "- First line: direct human-readable answer.\n"
            "- Then optional short explanation in plain English.\n"
            "- Do NOT output JSON.\n"
            "- Do NOT mention claim IDs or graph arrows.\n"
            "- If entity value starts with 'person_', show readable username without that prefix.\n\n"
            f"Question: {question}\n\n"
            "Claims:\n"
            + "\n".join(claim_lines)
            + "\n\nEvidence:\n"
            + "\n".join(evidence_lines)
        )

        try:
            resp = chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
            content = _extract_chat_content(resp)
            used_citations = sorted(set(re.findall(r"\[(E\d+)\]", content)))
            return {"answer": content, "used_citations": used_citations, "mode": "answer"}
        except Exception as e:
            return {
                "answer": f"Failed to generate answer with Ollama: {e}",
                "used_citations": [],
                "mode": "error",
            }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval + grounding from Neo4j memory graph")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument(
        "--mode",
        choices=["keyword", "hybrid"],
        default="keyword",
        help="Retrieval strategy: keyword or hybrid(keyword+embedding)",
    )
    parser.add_argument("--llm-model", default="llama3", help="Ollama model for answer generation")
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model for hybrid mode",
    )
    parser.add_argument("--max-claims-per-entity", type=int, default=10)
    parser.add_argument("--max-evidence-per-claim", type=int, default=3)
    parser.add_argument("--max-claims-total", type=int, default=30)
    parser.add_argument("--candidate-entity-limit", type=int, default=25)
    parser.add_argument("--skip-answer", action="store_true", help="Return context pack only")
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format. text gives a clean answer with citations.",
    )
    return parser.parse_args()


def _format_text_output(context_pack: Dict[str, Any]) -> str:
    if context_pack.get("error"):
        err = context_pack["error"]
        return f"Retrieval error: {err.get('type')}: {err.get('message')}"

    lines: List[str] = []
    answer = context_pack.get("answer")
    if isinstance(answer, dict):
        lines.append(str(answer.get("answer", "")).strip())
    else:
        lines.append("No answer generated.")

    evidence = context_pack.get("evidence", [])
    if evidence:
        lines.append("")
        lines.append("Evidence:")
        ev_by_id = {e.get("citation_id"): e for e in evidence if isinstance(e, dict)}
        used = []
        if isinstance(answer, dict):
            used = answer.get("used_citations") or []
        citations = used if used else [e.get("citation_id") for e in evidence]
        seen = set()
        for cid in citations:
            if cid in seen:
                continue
            seen.add(cid)
            ev = ev_by_id.get(cid)
            if not ev:
                continue
            excerpt = (ev.get("excerpt") or "").strip()
            source_type = ev.get("source_type")
            source_id = ev.get("source_id")
            url = ev.get("url")
            lines.append(f"[{cid}] {source_type} ({source_id})")
            if excerpt:
                lines.append(f"\"{excerpt}\"")
            if url:
                lines.append(f"URL: {url}")
            lines.append("")

    diagnostics = context_pack.get("diagnostics")
    if diagnostics and diagnostics.get("hint"):
        lines.append(diagnostics["hint"])

    return "\n".join(lines).strip()


def main() -> None:
    args = _parse_args()

    service = RetrievalGroundingService(
        mode=args.mode,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        max_claims_per_entity=args.max_claims_per_entity,
        max_evidence_per_claim=args.max_evidence_per_claim,
        max_claims_total=args.max_claims_total,
        candidate_entity_limit=args.candidate_entity_limit,
    )
    context_pack: Dict[str, Any]
    try:
        try:
            context_pack = service.retrieve_context_pack(args.question)
            if not args.skip_answer:
                context_pack["answer"] = service.generate_answer(context_pack)
        except Exception as e:
            context_pack = {
                "question": args.question,
                "retrieval_mode": args.mode,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }
    finally:
        service.close()

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(context_pack, f, indent=2, ensure_ascii=False)

    if args.format == "text":
        print(_format_text_output(context_pack))
    elif args.pretty:
        print(json.dumps(context_pack, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(context_pack, ensure_ascii=False))


if __name__ == "__main__":
    main()
