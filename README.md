# Airflow Issue Memory Graph
Builds a grounded memory graph from Apache Airflow GitHub issues, then retrieves and visualizes evidence-backed claims.

## 1. Project Title
**Airflow Issue Memory Graph**

One-line description: Extracts structured claims from issue data, stores them in Neo4j, retrieves grounded context packs, and visualizes entity/claim/evidence links in Cytoscape.js.

## 2. Project Overview
This system solves grounded memory retrieval for issue-tracking text.

What it builds:
- canonical entities (Issue/Person/Label/Status/Comment)
- timestamped claims between entities
- evidence snippets supporting each claim
- retrieval output with citations
- interactive graph view

High-level pipeline:
`Collection -> Cleaning -> Extraction -> Dedup -> Neo4j Graph -> Retrieval -> Visualization`

## 3. Dataset / Corpus
Dataset: **Apache Airflow GitHub Issues**.

Source:
- GitHub REST API (`apache/airflow` issues + comments)

How to reproduce dataset:
```bash
python3 Corpus/data_extraction.py
python3 Corpus/data_cleaning.py
```

Outputs:
- `data/raw/airflow_issues.json`
- `data/processed/airflow_clean.json`

## 4. System Architecture
Components:

Data Collection
↓
Cleaning
↓
Structured Extraction
↓
Deduplication
↓
Memory Graph (Neo4j)
↓
Retrieval (keyword / hybrid)
↓
Visualization (Cytoscape.js)

## 5. Ontology / Schema
### Entities
- `issue_{number}`
- `person_{login}`
- `comment_{id}`
- `label_kind_{name}` (or normalized label prefix)
- `status_{state}`

### Claims (predicates)
- `ISSUE_CREATED_BY`
- `ISSUE_HAS_LABEL`
- `ISSUE_HAS_STATUS`
- `ISSUE_HAS_COMMENT`
- `COMMENT_WRITTEN_BY`
- `ISSUE_ASSIGNED_TO` (LLM extractor)

### Evidence
Evidence fields include:
- `source_type`, `source_id`, `excerpt`, `timestamp`
- `offsets.char_start`, `offsets.char_end`
- reconstructed `url` during graph loading

Schemas:
- `data/claims/schema_v1.json`
- `data/Entities/schema_v1.json`
- `data/Entities/merge_log_schema_v1.json`

## 6. Memory Graph Design
Graph model:
- Node types: Entity, Claim, Evidence
- Edges:
  - `(Entity)-[:SUBJECT_OF]->(Claim)`
  - `(Claim)-[:OBJECT_OF]->(Entity)`
  - `(Claim)-[:SUPPORTED_BY]->(Evidence)`

Example:
`issue_62799 -[ISSUE_CREATED_BY]-> person_prajwal7842` with supporting evidence excerpt + URL.

## 7. Structured Extraction Pipeline
Deterministic extractor (`Extraction/claims.py`):
- metadata -> created_by/status/labels
- comment links -> issue_has_comment/comment_written_by

LLM extractor (`Extraction/extract_llm_claims.py`):
- strict JSON contract
- only `ISSUE_ASSIGNED_TO`
- only explicit `@username` mentions

Validation:
- schema validation (`jsonschema` if available, fallback checks)
- allowed subject/predicate checks

Claim IDs:
- UUIDv5 over `subject|predicate|object|event_time|source_id`

## 8. Deduplication & Canonicalization
Artifact dedup:
- normalized text hash (`sha256`) for issues/comments

Entity canonicalization:
- namespace normalization for person/label/status IDs
- alias/merge log support (`data/Entities/merge_log.json`)

Claim dedup:
- merge by `(subject,predicate,object)` (status also uses `valid_from`)
- aggregate evidence and support counts

Conflict handling:
- preserve temporal fields (`event_time`, `valid_from`, `valid_to`)

## 9. Temporal Modeling
- `event_time`: when source event happened
- `valid_from` / `valid_to`: validity interval
- “current” fact: `valid_to IS NULL` (or latest event in conflicts)

## 10. Retrieval System
`Retrieval/retrieval_grounding.py` supports:
- `keyword`: regex + entity matching
- `hybrid`: keyword + embedding rerank (Ollama embeddings)

Flow:
1. question -> candidate entities/predicate hints
2. fetch claims
3. fetch and rank evidence
4. return grounded context pack

Ranking uses confidence + recency + evidence support.

## 11. Context Pack Format
Example (trimmed):
```json
{
  "question": "Who created issue 62799?",
  "entities": [{"id": "issue_62799"}],
  "claims": [{
    "predicate": "ISSUE_CREATED_BY",
    "object": "person_prajwal7842",
    "citations": ["E1"]
  }],
  "evidence": [{
    "citation_id": "E1",
    "source_type": "issue_metadata",
    "source_id": 62799,
    "excerpt": "Logical date filter not working correctly...",
    "url": "https://github.com/apache/airflow/issues/62799"
  }]
}
```

## 12. Visualization Layer
Visualization uses Cytoscape.js static export:
- exporter: `Visualization/export_cytoscape_graph.py`
- output: `Visualization/cytoscape_graph.html`

UI features:
- graph exploration of entities/claims
- evidence panel on claim click
- clickable source URLs
- filters (type, predicate, time, confidence, merge mode)
- issue-scope control for de-cluttering

## 13. Evidence Grounding
Grounding is enforced at retrieval time:
- claims without evidence are dropped
- evidence carries excerpt + source metadata + URL
- offsets are stored when available
- comment links use issue-comment anchor reconstruction

## 14. Example Queries
1. `Who created issue 62799?`
2. `Who is assigned to issue 62887?`
3. `What labels are associated with issue 62903?`

Run example:
```bash
python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode keyword \
  --pretty
```

## 15. Running the Project (End-to-End)
### A. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests tqdm python-dotenv neo4j ollama streamlit jsonschema
```

### B. Services
1. Start Neo4j (Desktop or Docker) and create DB.
2. Start Ollama:
```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

### C. Configure `.env`
Set:
- `GITHUB_TOKEN`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- optional: `GITHUB_OWNER`, `GITHUB_REPO`

### D. Build data + graph
```bash
python3 Corpus/data_extraction.py
python3 Corpus/data_cleaning.py
python3 Extraction/claims.py
python3 Extraction/extract_llm_claims.py
python3 Extraction/claim_dedup.py
python3 Extraction/build_entities.py
python3 Graph_Design/load_entities_to_neo4j.py
python3 Graph_Design/load_claims_to_neo4j.py
python3 Graph_Design/load_evidence_to_neo4j.py
```

### E. Run retrieval
```bash
python3 Retrieval/retrieval_grounding.py --question "Who created issue 62799?" --mode keyword --pretty
python3 Retrieval/retrieval_grounding.py --question "Who created issue 62799?" --mode hybrid --pretty
```

### F. Launch visualization
```bash
python3 Visualization/export_cytoscape_graph.py
# Open Visualization/cytoscape_graph.html in browser
```

## 16. Repository Structure
```text
  Corpus/
  Extraction/
  Graph_Design/
  Retrieval/
  Visualization/
  data/
    raw/
    processed/
    claims/
    Entities/
```

## 17. Outputs Included
Generated outputs in repo:
- `data/claims/claims_v1.json`
- `data/claims/claims_llm_v1.json`
- `data/claims/claims_dedup_v1.json`
- `data/Entities/entities_v1.json`
- `Visualization/cytoscape_graph.html`

(If needed for submission, add screenshots and saved context-pack JSON files.)

## 18. Layer10 Adaptation Discussion
Same architecture adapts to:
- Email: thread/message as entities, claims from sender/mentions/replies
- Slack: channel/thread/message + user interactions
- Jira: ticket/comment/assignee/status transitions
- Internal docs: document/chunk/entity claims with citation spans

Core pattern remains: `Entity <-> Claim <-> Evidence` with temporal fields and grounded retrieval.

## 19. Limitations
- LLM extraction currently limited to assignment claims.
- Offset quality depends on exact/fuzzy text matching.
- Merge log may be sparse depending on current corpus slice.
- No production REST API wrapper yet (retrieval is script-first).

## 20. Future Improvements
- Add more predicates (mentions, PR/commit links).
- Add stronger entity resolution and merge confidence.
- Add REST API service with auth + caching.
- Add automated evaluation dashboard (precision/recall + latency).
- Add incremental ingestion and schema migration tooling.

---
If you are grading quickly: start with sections **15 (run)**, **5 (schema)**, **10 (retrieval)**, and **12 (visualization)**.
