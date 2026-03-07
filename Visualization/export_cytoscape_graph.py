#!/usr/bin/env python3
"""Export a static Cytoscape.js web graph for entities, claims, and evidence grounding."""

import json
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

ENTITIES_FILE = PROJECT_DIR / "data/Entities/entities_v1.json"
MERGE_LOG_FILE = PROJECT_DIR / "data/Entities/merge_log.json"
CLAIMS_CANDIDATES = [
    PROJECT_DIR / "data/claims/claims_dedup_v1.json",
    PROJECT_DIR / "data/claims/claims_merged_v1.json",
    PROJECT_DIR / "data/claims/claims_v1.json",
]
OUTPUT_HTML = SCRIPT_DIR / "cytoscape_graph.html"

GITHUB_REPO = "https://github.com/apache/airflow"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def choose_claims_file() -> Path:
    for path in CLAIMS_CANDIDATES:
        if path.exists():
            return path
    return CLAIMS_CANDIDATES[-1]


def issue_url(issue_id: str) -> str:
    return f"{GITHUB_REPO}/issues/{issue_id}"


def person_url(username: str) -> str:
    return f"https://github.com/{username}"


def build_comment_issue_map(claims: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for claim in claims:
        if claim.get("predicate") != "ISSUE_HAS_COMMENT":
            continue
        subject = claim.get("subject")
        obj = claim.get("object")
        if isinstance(subject, str) and subject.startswith("issue_") and isinstance(obj, str) and obj.startswith("comment_"):
            mapping[obj] = subject
    return mapping


def derive_entity_url(entity_id: str, entity_type: str, comment_to_issue: Dict[str, str]) -> str:
    if entity_id.startswith("issue_"):
        return issue_url(entity_id.split("issue_", 1)[1])
    if entity_id.startswith("person_"):
        username = entity_id.split("person_", 1)[1]
        if username:
            return person_url(username)
    if entity_id.startswith("comment_"):
        issue_entity = comment_to_issue.get(entity_id)
        comment_num = entity_id.split("comment_", 1)[1]
        if issue_entity and issue_entity.startswith("issue_") and comment_num:
            issue_num = issue_entity.split("issue_", 1)[1]
            return f"{issue_url(issue_num)}#issuecomment-{comment_num}"
    if entity_type == "Issue":
        suffix = entity_id.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return issue_url(suffix)
    return ""


def derive_evidence_url(
    evidence: Dict[str, Any],
    claim: Dict[str, Any],
    comment_to_issue: Dict[str, str],
    entity_url_index: Dict[str, str],
) -> str:
    source_type = (evidence.get("source_type") or "").strip()
    source_id = evidence.get("source_id")

    if source_type in {"issue_metadata", "issue_text"} and source_id is not None:
        return issue_url(str(source_id))

    if source_type == "issue_comment" and source_id is not None:
        comment_entity = f"comment_{source_id}"
        issue_entity = comment_to_issue.get(comment_entity)
        if issue_entity and issue_entity.startswith("issue_"):
            issue_num = issue_entity.split("issue_", 1)[1]
            return f"{issue_url(issue_num)}#issuecomment-{source_id}"

    subject = claim.get("subject")
    obj = claim.get("object")
    if isinstance(subject, str):
        fallback = entity_url_index.get(subject, "")
        if fallback:
            return fallback
    if isinstance(obj, str):
        fallback = entity_url_index.get(obj, "")
        if fallback:
            return fallback
    return ""


def pretty_entity_label(entity_id: str) -> str:
    if "_" not in entity_id:
        return entity_id
    prefix, suffix = entity_id.split("_", 1)
    if prefix == "person":
        return suffix
    return f"{prefix}:{suffix}"


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_graph_payload(
    entities: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    merge_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    comment_to_issue = build_comment_issue_map(claims)

    entity_index: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            continue
        entity_index[entity_id] = {
            "entity_id": entity_id,
            "entity_type": entity.get("entity_type", "Unknown"),
            "created_at": entity.get("created_at"),
            "last_seen_at": entity.get("last_seen_at"),
            "entity_version": entity.get("entity_version"),
            "aliases": entity.get("aliases") or [],
        }

    # Ensure all claim endpoints exist as nodes.
    for claim in claims:
        for key in ("subject", "object"):
            entity_id = claim.get(key)
            if not isinstance(entity_id, str) or not entity_id:
                continue
            if entity_id not in entity_index:
                entity_index[entity_id] = {
                    "entity_id": entity_id,
                    "entity_type": "Unknown",
                    "created_at": claim.get("event_time"),
                    "last_seen_at": claim.get("event_time"),
                    "entity_version": "1.0",
                    "aliases": [],
                }

    entity_merge_index: Dict[str, List[Dict[str, Any]]] = {}
    for row in merge_log:
        target = row.get("target_entity")
        if not isinstance(target, str) or not target:
            continue
        entity_merge_index.setdefault(target, []).append(
            {
                "merge_id": row.get("merge_id"),
                "source_entity": row.get("source_entity"),
                "reason": row.get("reason"),
                "timestamp": row.get("timestamp"),
            }
        )

    entity_url_index: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []

    for entity_id, entity in sorted(entity_index.items(), key=lambda x: x[0]):
        entity_type = entity.get("entity_type", "Unknown")
        url = derive_entity_url(entity_id, entity_type, comment_to_issue)
        entity_url_index[entity_id] = url
        merges = entity_merge_index.get(entity_id, [])

        node_data = {
            "id": entity_id,
            "label": pretty_entity_label(entity_id),
            "entity_id": entity_id,
            "entity_type": entity_type,
            "url": url,
            "created_at": entity.get("created_at"),
            "last_seen_at": entity.get("last_seen_at"),
            "entity_version": entity.get("entity_version"),
            "aliases": entity.get("aliases") or [],
            "alias_count": len(entity.get("aliases") or []),
            "merge_entries": merges,
            "merge_count": len(merges),
            "has_merge_signal": bool((entity.get("aliases") or []) or merges),
        }
        node_classes = ["entity", f"type-{str(entity_type).lower()}"]
        nodes.append({"data": node_data, "classes": " ".join(node_classes)})

    edges: List[Dict[str, Any]] = []
    merged_claim_rows: List[Dict[str, Any]] = []

    for idx, claim in enumerate(claims):
        subject = claim.get("subject")
        obj = claim.get("object")
        predicate = claim.get("predicate")
        claim_id = claim.get("claim_id") or f"claim_{idx}"
        if not (isinstance(subject, str) and isinstance(obj, str) and isinstance(predicate, str)):
            continue

        evidences = claim.get("supporting_evidence")
        if not isinstance(evidences, list) or not evidences:
            single_ev = claim.get("evidence")
            evidences = [single_ev] if isinstance(single_ev, dict) else []

        evidence_rows: List[Dict[str, Any]] = []
        for i, ev in enumerate(evidences):
            if not isinstance(ev, dict):
                continue
            excerpt = (ev.get("excerpt") or "").strip()
            evidence_rows.append(
                {
                    "evidence_id": f"{claim_id}_e{i+1}",
                    "source_type": ev.get("source_type"),
                    "source_id": ev.get("source_id"),
                    "excerpt": excerpt,
                    "timestamp": ev.get("timestamp"),
                    "location": ev.get("location"),
                    "offsets": ev.get("offsets") or {},
                    "url": derive_evidence_url(ev, claim, comment_to_issue, entity_url_index),
                    "has_excerpt": bool(excerpt),
                }
            )

        support_count = int(claim.get("support_count") or len(evidence_rows) or 1)
        merged_claim = support_count > 1 or len(evidence_rows) > 1

        if merged_claim:
            merged_claim_rows.append(
                {
                    "claim_id": claim_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "support_count": support_count,
                    "evidence_count": len(evidence_rows),
                }
            )

        edge_data = {
            "id": f"claim::{claim_id}",
            "claim_id": claim_id,
            "source": subject,
            "target": obj,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": round(as_float(claim.get("confidence"), 0.0), 4),
            "event_time": claim.get("event_time"),
            "valid_from": claim.get("valid_from"),
            "valid_to": claim.get("valid_to"),
            "support_count": support_count,
            "evidence_count": len(evidence_rows),
            "merged_claim": merged_claim,
            "claim_url": "",
            "evidence": evidence_rows,
        }

        edges.append(
            {
                "data": edge_data,
                "classes": "claim-edge merged-claim" if merged_claim else "claim-edge",
            }
        )

    entities_with_aliases = [
        {
            "entity_id": n["data"]["entity_id"],
            "entity_type": n["data"]["entity_type"],
            "alias_count": n["data"]["alias_count"],
            "aliases": n["data"]["aliases"],
        }
        for n in nodes
        if n["data"]["alias_count"] > 0
    ]

    entities_with_merge_log = [
        {
            "entity_id": n["data"]["entity_id"],
            "entity_type": n["data"]["entity_type"],
            "merge_count": n["data"]["merge_count"],
            "merge_entries": n["data"]["merge_entries"],
        }
        for n in nodes
        if n["data"]["merge_count"] > 0
    ]

    predicate_values = sorted({e["data"]["predicate"] for e in edges})
    entity_type_values = sorted({n["data"]["entity_type"] for n in nodes})

    return {
        "elements": {"nodes": nodes, "edges": edges},
        "meta": {
            "entity_count": len(nodes),
            "claim_count": len(edges),
            "predicate_values": predicate_values,
            "entity_type_values": entity_type_values,
            "source_claim_file": str(choose_claims_file().relative_to(PROJECT_DIR)),
        },
        "merges": {
            "entities_with_aliases": sorted(entities_with_aliases, key=lambda x: x["alias_count"], reverse=True),
            "entities_with_merge_log": sorted(entities_with_merge_log, key=lambda x: x["merge_count"], reverse=True),
            "merged_claims": sorted(merged_claim_rows, key=lambda x: x["support_count"], reverse=True),
        },
    }


HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Memory Graph Explorer (Cytoscape.js)</title>
  <style>
    :root {
      --bg: #f4f7f9;
      --panel: #ffffff;
      --panel-2: #f9fbfc;
      --border: #d8e2e8;
      --text: #1e2933;
      --muted: #60717d;
      --accent: #0b6e99;
      --accent-soft: #dff0f7;
      --ok: #1f7a1f;
      --warn: #a35f00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      display: grid;
      grid-template-rows: auto 1fr;
      height: 100vh;
    }
    .top {
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #ffffff 0%, #f5f9fb 100%);
      padding: 12px 14px;
      display: grid;
      gap: 10px;
    }
    .title-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .title-row h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(9, minmax(120px, 1fr));
      gap: 8px;
      align-items: end;
    }
    .controls .field {
      display: grid;
      gap: 4px;
      min-width: 0;
    }
    label { font-size: 12px; color: var(--muted); }
    input, select, button {
      width: 100%;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 7px 9px;
      font-size: 13px;
    }
    select[multiple] {
      min-height: 110px;
      line-height: 1.2;
    }
    .select-actions {
      display: flex;
      gap: 6px;
      margin-top: 4px;
    }
    .select-actions button {
      width: auto;
      padding: 4px 8px;
      font-size: 12px;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 600;
    }
    button.secondary {
      background: #fff;
      color: var(--text);
      border-color: var(--border);
    }
    .content {
      min-height: 0;
      display: grid;
      grid-template-columns: 1fr 400px;
      gap: 0;
    }
    #cy {
      min-width: 0;
      min-height: 0;
      border-right: 1px solid var(--border);
      background: radial-gradient(circle at 40% 30%, #ffffff 0%, #edf4f7 75%);
    }
    .side {
      min-height: 0;
      overflow: auto;
      background: var(--panel);
      padding: 12px;
      display: grid;
      gap: 12px;
    }
    .card {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel-2);
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .card h2 {
      margin: 0;
      font-size: 14px;
    }
    .muted { color: var(--muted); }
    .kv {
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 6px;
      font-size: 13px;
      align-items: start;
    }
    .badge {
      display: inline-block;
      padding: 3px 7px;
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 11px;
      margin-right: 5px;
      margin-bottom: 5px;
      background: #fff;
    }
    .evidence {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      display: grid;
      gap: 6px;
      font-size: 13px;
    }
    .excerpt {
      background: #f5fafc;
      border: 1px solid #d8ecf5;
      border-radius: 8px;
      padding: 8px;
      white-space: pre-wrap;
      line-height: 1.35;
    }
    .merge-row {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      padding: 7px;
      font-size: 12px;
      display: grid;
      gap: 5px;
    }
    .pill-btn {
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #fff;
      padding: 3px 8px;
      margin: 2px;
      font-size: 12px;
      cursor: pointer;
      width: auto;
    }
    .pill-btn:hover { border-color: var(--accent); color: var(--accent); }
    .warn { color: var(--warn); }
    .ok { color: var(--ok); }
    .small { font-size: 12px; }
    @media (max-width: 1100px) {
      .content { grid-template-columns: 1fr; }
      #cy { min-height: 48vh; border-right: none; border-bottom: 1px solid var(--border); }
      .controls { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class=\"top\">
    <div class=\"title-row\">
      <h1>Memory Graph Explorer</h1>
      <div id=\"metaLine\" class=\"meta\"></div>
    </div>
    <div class=\"controls\">
      <div class=\"field\">
        <label for=\"entityTypes\">Entity Types</label>
        <select id=\"entityTypes\" multiple size=\"4\"></select>
        <div class=\"select-actions\">
          <button id=\"entityTypesAll\" class=\"secondary\" type=\"button\">All</button>
          <button id=\"entityTypesNone\" class=\"secondary\" type=\"button\">None</button>
        </div>
      </div>
      <div class=\"field\">
        <label for=\"predicates\">Claim Predicates</label>
        <select id=\"predicates\" multiple size=\"4\"></select>
        <div class=\"select-actions\">
          <button id=\"predicatesAll\" class=\"secondary\" type=\"button\">All</button>
          <button id=\"predicatesNone\" class=\"secondary\" type=\"button\">None</button>
        </div>
      </div>
      <div class=\"field\">
        <label for=\"maxIssues\">Max Issues (with neighbors)</label>
        <input id=\"maxIssues\" type=\"number\" min=\"0\" step=\"1\" value=\"30\" />
        <div class=\"small muted\">0 = show all issues</div>
      </div>
      <div class=\"field\">
        <label for=\"minConfidence\">Min Confidence</label>
        <input id=\"minConfidence\" type=\"range\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0\" />
        <div id=\"confidenceVal\" class=\"small muted\">0.00</div>
      </div>
      <div class=\"field\">
        <label for=\"dateFrom\">Event Time From</label>
        <input id=\"dateFrom\" type=\"date\" />
      </div>
      <div class=\"field\">
        <label for=\"dateTo\">Event Time To</label>
        <input id=\"dateTo\" type=\"date\" />
      </div>
      <div class=\"field\">
        <label for=\"mergedOnly\">Merge Filter</label>
        <select id=\"mergedOnly\">
          <option value=\"any\">All Claims</option>
          <option value=\"merged\">Merged Only</option>
          <option value=\"single\">Single-support Only</option>
        </select>
      </div>
      <div class=\"field\">
        <label for=\"search\">Find Entity / Claim</label>
        <input id=\"search\" type=\"text\" placeholder=\"issue_62799 or claim id\" />
      </div>
      <div class=\"field\">
        <label>&nbsp;</label>
        <div style=\"display:flex; gap:8px;\">
          <button id=\"searchBtn\">Find</button>
          <button id=\"resetBtn\" class=\"secondary\">Reset</button>
        </div>
      </div>
    </div>
  </div>

  <div class=\"content\">
    <div id=\"cy\"></div>
    <div class=\"side\">
      <div class=\"card\" id=\"selectionPanel\">
        <h2>Selection</h2>
        <div class=\"muted small\">Click any entity node or claim edge to inspect evidence and metadata.</div>
      </div>

      <div class=\"card\" id=\"mergePanel\">
        <h2>Duplicates / Merges</h2>
        <div id=\"mergeSummary\" class=\"small muted\"></div>
        <div>
          <div class=\"small\"><strong>Entities With Aliases</strong></div>
          <div id=\"aliasList\"></div>
        </div>
        <div>
          <div class=\"small\"><strong>Entity Merge Log</strong></div>
          <div id=\"entityMergeList\"></div>
        </div>
        <div>
          <div class=\"small\"><strong>Merged Claims</strong></div>
          <div id=\"claimMergeList\"></div>
        </div>
      </div>
    </div>
  </div>

  <script src=\"https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js\"></script>
  <script>
    const PAYLOAD = __PAYLOAD__;

    if (!window.cytoscape) {
      document.body.innerHTML = '<div style="padding:20px;font-family:Segoe UI,sans-serif">Cytoscape.js failed to load. Open this file with internet access or bundle cytoscape locally.</div>';
      throw new Error('Cytoscape unavailable');
    }

    const entityTypesEl = document.getElementById('entityTypes');
    const entityTypesAllEl = document.getElementById('entityTypesAll');
    const entityTypesNoneEl = document.getElementById('entityTypesNone');
    const predicatesEl = document.getElementById('predicates');
    const predicatesAllEl = document.getElementById('predicatesAll');
    const predicatesNoneEl = document.getElementById('predicatesNone');
    const maxIssuesEl = document.getElementById('maxIssues');
    const minConfidenceEl = document.getElementById('minConfidence');
    const confidenceValEl = document.getElementById('confidenceVal');
    const dateFromEl = document.getElementById('dateFrom');
    const dateToEl = document.getElementById('dateTo');
    const mergedOnlyEl = document.getElementById('mergedOnly');
    const searchEl = document.getElementById('search');
    const searchBtnEl = document.getElementById('searchBtn');
    const resetBtnEl = document.getElementById('resetBtn');
    const metaLineEl = document.getElementById('metaLine');

    const selectionPanel = document.getElementById('selectionPanel');
    const mergeSummary = document.getElementById('mergeSummary');
    const aliasList = document.getElementById('aliasList');
    const entityMergeList = document.getElementById('entityMergeList');
    const claimMergeList = document.getElementById('claimMergeList');

    const EDGE_COLOR = {
      ISSUE_CREATED_BY: '#1f78b4',
      ISSUE_ASSIGNED_TO: '#3f8f3f',
      ISSUE_HAS_STATUS: '#f39c12',
      ISSUE_HAS_LABEL: '#8e44ad',
      ISSUE_HAS_COMMENT: '#16a085',
      COMMENT_WRITTEN_BY: '#e74c3c'
    };

    function selectedValues(selectEl) {
      return Array.from(selectEl.options)
        .filter((o) => o.selected)
        .map((o) => o.value);
    }

    function setSelectState(selectEl, selected) {
      Array.from(selectEl.options).forEach((opt) => {
        opt.selected = selected;
      });
    }

    function setMultiOptions(selectEl, values) {
      selectEl.innerHTML = '';
      values.forEach((v) => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        opt.selected = true;
        selectEl.appendChild(opt);
      });
    }

    function toDateOrNull(ts) {
      if (!ts || typeof ts !== 'string') return null;
      const d = new Date(ts);
      return Number.isNaN(d.getTime()) ? null : d;
    }

    function parseIssueNumber(entityId) {
      if (typeof entityId !== 'string' || !entityId.startsWith('issue_')) return -1;
      const value = Number(entityId.slice(6));
      return Number.isFinite(value) ? value : -1;
    }

    function esc(str) {
      return String(str ?? '').replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
    }

    setMultiOptions(entityTypesEl, PAYLOAD.meta.entity_type_values);
    setMultiOptions(predicatesEl, PAYLOAD.meta.predicate_values);

    metaLineEl.textContent = `${PAYLOAD.meta.entity_count} entities | ${PAYLOAD.meta.claim_count} claims | source: ${PAYLOAD.meta.source_claim_file}`;

    const cy = cytoscape({
      container: document.getElementById('cy'),
      elements: [...PAYLOAD.elements.nodes, ...PAYLOAD.elements.edges],
      wheelSensitivity: 0.15,
      layout: { name: 'cose', animate: false, fit: true, padding: 24, idealEdgeLength: 120 },
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#95a5a6',
            'label': 'data(label)',
            'font-size': 10,
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': 70,
            'color': '#102028',
            'width': 28,
            'height': 28,
            'border-width': 1,
            'border-color': '#35505f'
          }
        },
        { selector: 'node[type-issue]', style: { 'background-color': '#72aee6' } },
        { selector: 'node[type-person]', style: { 'background-color': '#7ecf8a' } },
        { selector: 'node[type-comment]', style: { 'background-color': '#f4b183' } },
        { selector: 'node[type-label]', style: { 'background-color': '#d3b6f0' } },
        { selector: 'node[type-status]', style: { 'background-color': '#f6d06f' } },
        { selector: 'node[type-unknown]', style: { 'background-color': '#c4c4c4' } },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'line-color': '#9fb3bf',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#9fb3bf',
            'width': 2,
            'opacity': 0.85
          }
        },
        {
          selector: 'edge.merged-claim',
          style: {
            'line-style': 'dashed',
            'width': 3
          }
        },
        {
          selector: 'edge:selected, node:selected',
          style: {
            'border-width': 3,
            'border-color': '#0b6e99',
            'line-color': '#0b6e99',
            'target-arrow-color': '#0b6e99',
            'z-index': 999
          }
        }
      ]
    });

    cy.edges().forEach((edge) => {
      const pred = edge.data('predicate');
      edge.style('line-color', EDGE_COLOR[pred] || '#9fb3bf');
      edge.style('target-arrow-color', EDGE_COLOR[pred] || '#9fb3bf');
    });

    function renderSelection(html) {
      selectionPanel.innerHTML = `<h2>Selection</h2>${html}`;
    }

    function evidenceCard(ev, idx) {
      const excerpt = ev.excerpt ? `<div class=\"excerpt\">${esc(ev.excerpt)}</div>` : '<div class="muted">No excerpt</div>';
      const url = ev.url ? `<a href=\"${esc(ev.url)}\" target=\"_blank\" rel=\"noopener\">${esc(ev.url)}</a>` : '<span class="muted">(no url)</span>';
      return `
        <div class=\"evidence\">
          <div><strong>[E${idx + 1}]</strong> ${esc(ev.source_type || 'unknown')} (${esc(ev.source_id || '')})</div>
          ${excerpt}
          <div class=\"small\"><strong>timestamp:</strong> ${esc(ev.timestamp || '')}</div>
          <div class=\"small\"><strong>url:</strong> ${url}</div>
        </div>
      `;
    }

    function showNodeDetails(node) {
      const d = node.data();
      const aliases = (d.aliases || []).length
        ? d.aliases.map((a) => `<span class=\"badge\">${esc(a)}</span>`).join('')
        : '<span class="muted">None</span>';
      const mergeRows = (d.merge_entries || []).length
        ? d.merge_entries
            .map(
              (m) => `<div class=\"merge-row\"><div><strong>source:</strong> ${esc(m.source_entity)}</div><div><strong>reason:</strong> ${esc(m.reason)}</div><div><strong>time:</strong> ${esc(m.timestamp)}</div></div>`
            )
            .join('')
        : '<span class="muted">None</span>';
      const url = d.url ? `<a href=\"${esc(d.url)}\" target=\"_blank\" rel=\"noopener\">${esc(d.url)}</a>` : '<span class="muted">(no url)</span>';

      renderSelection(`
        <div class=\"kv\"><div>entity_id</div><div><code>${esc(d.entity_id)}</code></div></div>
        <div class=\"kv\"><div>entity_type</div><div>${esc(d.entity_type)}</div></div>
        <div class=\"kv\"><div>created_at</div><div>${esc(d.created_at || '')}</div></div>
        <div class=\"kv\"><div>last_seen_at</div><div>${esc(d.last_seen_at || '')}</div></div>
        <div class=\"kv\"><div>url</div><div>${url}</div></div>
        <div><strong>aliases</strong><div>${aliases}</div></div>
        <div><strong>merge entries</strong><div>${mergeRows}</div></div>
      `);
    }

    function showEdgeDetails(edge) {
      const d = edge.data();
      const evRows = (d.evidence || []).map((ev, idx) => evidenceCard(ev, idx)).join('');
      const mergedBadge = d.merged_claim ? '<span class="badge">merged-claim</span>' : '';

      renderSelection(`
        <div class=\"kv\"><div>claim_id</div><div><code>${esc(d.claim_id)}</code></div></div>
        <div class=\"kv\"><div>triple</div><div><code>${esc(d.subject)}</code> -[<code>${esc(d.predicate)}</code>]-> <code>${esc(d.object)}</code></div></div>
        <div class=\"kv\"><div>confidence</div><div>${esc(d.confidence)}</div></div>
        <div class=\"kv\"><div>event_time</div><div>${esc(d.event_time || '')}</div></div>
        <div class=\"kv\"><div>valid_from</div><div>${esc(d.valid_from || '')}</div></div>
        <div class=\"kv\"><div>valid_to</div><div>${esc(d.valid_to || '')}</div></div>
        <div class=\"kv\"><div>support_count</div><div>${esc(d.support_count)} ${mergedBadge}</div></div>
        <div><strong>Evidence (${(d.evidence || []).length})</strong></div>
        <div>${evRows || '<div class="warn">No evidence attached.</div>'}</div>
      `);
    }

    function dateRangePass(edgeDate, fromDate, toDate) {
      if (!edgeDate) return true;
      if (fromDate && edgeDate < fromDate) return false;
      if (toDate && edgeDate > toDate) return false;
      return true;
    }

    function scopeEdgesToIssues(candidateEdges, maxIssues) {
      if (!Number.isFinite(maxIssues) || maxIssues <= 0) {
        return {
          edgeIds: new Set(candidateEdges.map((e) => e.id())),
          nodeIds: null,
          scopedIssueCount: 0
        };
      }

      const issueNodes = cy.nodes()
        .toArray()
        .filter((n) => n.data('entity_type') === 'Issue')
        .sort((a, b) => {
          const ad = toDateOrNull(a.data('created_at'));
          const bd = toDateOrNull(b.data('created_at'));
          if (ad && bd && ad.getTime() !== bd.getTime()) return bd.getTime() - ad.getTime();
          if (ad && !bd) return -1;
          if (!ad && bd) return 1;
          return parseIssueNumber(b.id()) - parseIssueNumber(a.id());
        });

      const scopedIssues = issueNodes.slice(0, maxIssues).map((n) => n.id());
      const scopedIssueSet = new Set(scopedIssues);

      const stage1 = candidateEdges.filter((edge) => {
        const sourceId = edge.data('source');
        const targetId = edge.data('target');
        return scopedIssueSet.has(sourceId) || scopedIssueSet.has(targetId);
      });

      const nodeIds = new Set(scopedIssues);
      stage1.forEach((edge) => {
        nodeIds.add(edge.data('source'));
        nodeIds.add(edge.data('target'));
      });

      const stage2 = candidateEdges.filter((edge) => {
        const sourceId = edge.data('source');
        const targetId = edge.data('target');
        return nodeIds.has(sourceId) && nodeIds.has(targetId);
      });

      const edgeIds = new Set();
      [...stage1, ...stage2].forEach((edge) => {
        edgeIds.add(edge.id());
        nodeIds.add(edge.data('source'));
        nodeIds.add(edge.data('target'));
      });

      return {
        edgeIds,
        nodeIds,
        scopedIssueCount: scopedIssues.length
      };
    }

    function applyFilters() {
      const selectedTypes = new Set(selectedValues(entityTypesEl));
      const selectedPreds = new Set(selectedValues(predicatesEl));
      const minConfidence = Number(minConfidenceEl.value || 0);
      const mergedOnly = mergedOnlyEl.value;
      let maxIssues = Number.parseInt(maxIssuesEl.value || '0', 10);
      if (!Number.isFinite(maxIssues) || maxIssues < 0) maxIssues = 0;
      maxIssuesEl.value = String(maxIssues);
      confidenceValEl.textContent = minConfidence.toFixed(2);

      const fromDate = dateFromEl.value ? new Date(dateFromEl.value + 'T00:00:00Z') : null;
      const toDate = dateToEl.value ? new Date(dateToEl.value + 'T23:59:59Z') : null;

      if (selectedTypes.size === 0 || selectedPreds.size === 0) {
        cy.batch(() => {
          cy.edges().forEach((edge) => edge.style('display', 'none'));
          cy.nodes().forEach((node) => node.style('display', 'none'));
        });
        metaLineEl.textContent = `0/${PAYLOAD.meta.entity_count} entities | 0/${PAYLOAD.meta.claim_count} claims | source: ${PAYLOAD.meta.source_claim_file}`;
        return;
      }

      const nodeTypePass = new Map();
      cy.nodes().forEach((node) => {
        nodeTypePass.set(node.id(), selectedTypes.has(node.data('entity_type')));
      });

      const candidateEdges = [];
      cy.edges().forEach((edge) => {
        const pred = edge.data('predicate');
        const conf = Number(edge.data('confidence') || 0);
        const edgeDate = toDateOrNull(edge.data('event_time'));
        const isMerged = Boolean(edge.data('merged_claim'));
        const sourceId = edge.data('source');
        const targetId = edge.data('target');

        let visible = selectedPreds.has(pred) && conf >= minConfidence && dateRangePass(edgeDate, fromDate, toDate);
        visible = visible && Boolean(nodeTypePass.get(sourceId)) && Boolean(nodeTypePass.get(targetId));
        if (mergedOnly === 'merged' && !isMerged) visible = false;
        if (mergedOnly === 'single' && isMerged) visible = false;

        if (visible) {
          candidateEdges.push(edge);
        }
      });

      const scoped = scopeEdgesToIssues(candidateEdges, maxIssues);
      const visibleEdgeIds = scoped.edgeIds;
      const visibleNodeIds = scoped.nodeIds ? new Set(scoped.nodeIds) : new Set();

      cy.batch(() => {
        cy.edges().forEach((edge) => {
          const isVisible = visibleEdgeIds.has(edge.id());
          edge.style('display', isVisible ? 'element' : 'none');
          if (isVisible) {
            visibleNodeIds.add(edge.data('source'));
            visibleNodeIds.add(edge.data('target'));
          }
        });

        cy.nodes().forEach((node) => {
          const isVisible = Boolean(nodeTypePass.get(node.id())) && visibleNodeIds.has(node.id());
          node.style('display', isVisible ? 'element' : 'none');
        });
      });

      const visibleNodes = cy.nodes(':visible').length;
      const visibleEdges = cy.edges(':visible').length;
      const scopeLabel = maxIssues > 0 ? ` | issue scope: ${scoped.scopedIssueCount}` : ' | issue scope: all';
      metaLineEl.textContent = `${visibleNodes}/${PAYLOAD.meta.entity_count} entities | ${visibleEdges}/${PAYLOAD.meta.claim_count} claims${scopeLabel} | source: ${PAYLOAD.meta.source_claim_file}`;
    }

    function focusBySearch() {
      const q = (searchEl.value || '').trim().toLowerCase();
      if (!q) return;

      const node = cy.nodes().filter((n) => {
        const id = String(n.data('id') || '').toLowerCase();
        const label = String(n.data('label') || '').toLowerCase();
        return id === q || id.includes(q) || label.includes(q);
      }).first();

      if (node && node.length > 0) {
        cy.elements().unselect();
        node.select();
        cy.animate({ center: { eles: node }, zoom: 1.2, duration: 350 });
        showNodeDetails(node);
        return;
      }

      const edge = cy.edges().filter((e) => {
        const claimId = String(e.data('claim_id') || '').toLowerCase();
        return claimId === q || claimId.includes(q);
      }).first();

      if (edge && edge.length > 0) {
        cy.elements().unselect();
        edge.select();
        cy.animate({ fit: { eles: edge.connectedNodes().union(edge), padding: 80 }, duration: 350 });
        showEdgeDetails(edge);
      }
    }

    function resetFilters() {
      setSelectState(entityTypesEl, true);
      setSelectState(predicatesEl, true);
      maxIssuesEl.value = '30';
      minConfidenceEl.value = '0';
      dateFromEl.value = '';
      dateToEl.value = '';
      mergedOnlyEl.value = 'any';
      searchEl.value = '';
      applyFilters();
      const visibleElements = cy.elements(':visible');
      if (visibleElements.length > 0) {
        cy.fit(visibleElements, 30);
      }
    }

    function makePill(label, onClick) {
      const btn = document.createElement('button');
      btn.className = 'pill-btn';
      btn.textContent = label;
      btn.onclick = onClick;
      return btn;
    }

    function renderMergePanels() {
      const aliases = PAYLOAD.merges.entities_with_aliases || [];
      const entityMerges = PAYLOAD.merges.entities_with_merge_log || [];
      const claimMerges = PAYLOAD.merges.merged_claims || [];

      mergeSummary.innerHTML = `
        <div>${aliases.length} entities have aliases.</div>
        <div>${entityMerges.length} entities in merge log.</div>
        <div>${claimMerges.length} claims are merged (support_count > 1).</div>
      `;

      aliasList.innerHTML = '';
      aliases.slice(0, 50).forEach((row) => {
        aliasList.appendChild(makePill(`${row.entity_id} (${row.alias_count})`, () => {
          const n = cy.getElementById(row.entity_id);
          if (n && n.length) {
            cy.elements().unselect();
            n.select();
            cy.animate({ center: { eles: n }, zoom: 1.25, duration: 300 });
            showNodeDetails(n);
          }
        }));
      });
      if (!aliases.length) aliasList.innerHTML = '<span class="muted small">None</span>';

      entityMergeList.innerHTML = '';
      entityMerges.slice(0, 50).forEach((row) => {
        entityMergeList.appendChild(makePill(`${row.entity_id} (${row.merge_count})`, () => {
          const n = cy.getElementById(row.entity_id);
          if (n && n.length) {
            cy.elements().unselect();
            n.select();
            cy.animate({ center: { eles: n }, zoom: 1.25, duration: 300 });
            showNodeDetails(n);
          }
        }));
      });
      if (!entityMerges.length) entityMergeList.innerHTML = '<span class="muted small">merge_log.json is empty</span>';

      claimMergeList.innerHTML = '';
      claimMerges.slice(0, 80).forEach((row) => {
        claimMergeList.appendChild(makePill(`${row.claim_id} (${row.support_count})`, () => {
          const e = cy.getElementById(`claim::${row.claim_id}`);
          if (e && e.length) {
            cy.elements().unselect();
            e.select();
            cy.animate({ fit: { eles: e.connectedNodes().union(e), padding: 80 }, duration: 300 });
            showEdgeDetails(e);
          }
        }));
      });
      if (!claimMerges.length) claimMergeList.innerHTML = '<span class="muted small">No merged claims found.</span>';
    }

    cy.on('tap', 'node', (evt) => showNodeDetails(evt.target));
    cy.on('tap', 'edge', (evt) => showEdgeDetails(evt.target));
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        renderSelection('<div class="muted small">Click any entity node or claim edge to inspect evidence and metadata.</div>');
      }
    });

    entityTypesAllEl.addEventListener('click', () => {
      setSelectState(entityTypesEl, true);
      applyFilters();
    });
    entityTypesNoneEl.addEventListener('click', () => {
      setSelectState(entityTypesEl, false);
      applyFilters();
    });
    predicatesAllEl.addEventListener('click', () => {
      setSelectState(predicatesEl, true);
      applyFilters();
    });
    predicatesNoneEl.addEventListener('click', () => {
      setSelectState(predicatesEl, false);
      applyFilters();
    });

    [entityTypesEl, predicatesEl, maxIssuesEl, minConfidenceEl, dateFromEl, dateToEl, mergedOnlyEl].forEach((el) => {
      el.addEventListener('change', applyFilters);
      el.addEventListener('input', applyFilters);
    });

    searchBtnEl.addEventListener('click', focusBySearch);
    searchEl.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') focusBySearch();
    });
    resetBtnEl.addEventListener('click', resetFilters);

    renderMergePanels();
    applyFilters();
  </script>
</body>
</html>
"""


def write_html(payload: Dict[str, Any], output_path: Path) -> None:
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    claims_path = choose_claims_file()
    entities = load_json(ENTITIES_FILE, [])
    claims = load_json(claims_path, [])
    merge_log = load_json(MERGE_LOG_FILE, [])

    if not isinstance(entities, list):
        raise SystemExit(f"Invalid entities json: {ENTITIES_FILE}")
    if not isinstance(claims, list):
        raise SystemExit(f"Invalid claims json: {claims_path}")
    if not isinstance(merge_log, list):
        merge_log = []

    payload = build_graph_payload(entities, claims, merge_log)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    write_html(payload, OUTPUT_HTML)

    print(f"Wrote {OUTPUT_HTML}")
    print(
        f"Entities: {payload['meta']['entity_count']} | "
        f"Claims: {payload['meta']['claim_count']} | "
        f"Source claims: {payload['meta']['source_claim_file']}"
    )


if __name__ == "__main__":
    main()
