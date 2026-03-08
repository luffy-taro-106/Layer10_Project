# Airflow Issue Memory Graph
Builds a grounded memory graph from Apache Airflow GitHub issues, stores it in Neo4j, supports grounded retrieval, and exports an interactive Cytoscape.js graph.

## Author

**Darshan Sonawane 23B1541**  

Project: Layer10 Memory Graph

## What This Project Does
- Collects GitHub issues + comments from `apache/airflow`
- Cleans and normalizes raw artifacts
- Extracts structured claims (deterministic + LLM assignment claims)
- Deduplicates claims and builds canonical entities
- Loads Entity/Claim/Evidence graph into Neo4j
- Retrieves grounded context packs for user questions
- Exports a static interactive web graph

## High-Level Architecture
```text
GitHub API
  -> Corpus cleaning
  -> Claim extraction (deterministic + LLM)
  -> Claim dedup + entity build
  -> Neo4j ingestion (entities, claims, evidence)
  -> Retrieval (keyword/hybrid)
  -> Visualization (Cytoscape.js)
```

## Submission Artifacts
- Write-up PDF: `Layer10_Writeup.pdf`
- Serialized memory graph export: `memory_graph.json`
- Example context packs:
  - `context_packs_examples/context_pack_q1_issue_62799_creator.json`
  - `context_packs_examples/context_pack_q2_issue_62887_assignee.json`
  - `context_packs_examples/context_pack_q3_issue_62903_labels.json`

## Repository Layout
```text
Corpus/           # data extraction + cleaning
Extraction/       # claims, llm extraction, dedup, entities
Graph_Design/     # Neo4j loaders
Retrieval/        # retrieval + grounding scripts/UI
Visualization/    # Cytoscape export
data/
  raw/
  processed/
  claims/
  Entities/
```

## Prerequisites
- Python 3.10+
- Neo4j instance (local or Aura)
- Ollama (for LLM extraction/retrieval answer generation)
- GitHub token

## Setup
From repo root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create env file:
```bash
cp .env.example .env
```

Set required values in `.env`:
- `GITHUB_TOKEN`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- optional: `GITHUB_OWNER`, `GITHUB_REPO`

Start Ollama models:
```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

## Run End-to-End
Run all pipeline steps in order:

```bash
source .venv/bin/activate

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

## Retrieval
Keyword mode:
```bash
python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode keyword \
  --pretty
```

Hybrid mode (keyword + embeddings):
```bash
python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode hybrid \
  --embedding-model nomic-embed-text \
  --pretty
```

Save context pack JSON:
```bash
python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode keyword \
  --skip-answer \
  --format json \
  --pretty \
  --output context_packs_examples/context_pack_q1_issue_62799_creator.json
```

Optional Streamlit UI:
```bash
streamlit run Retrieval/app.py
```

## Visualization
Export static interactive graph:
```bash
python3 Visualization/export_cytoscape_graph.py
```
Open:
- `Visualization/cytoscape_graph.html`

## Get Serialized Output of the Graph

Run the following Cypher query in the Neo4j Browser to retrieve a serialized representation of the memory graph:

```cypher
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN
collect(DISTINCT n) AS nodes,
collect(DISTINCT r) AS relationships,
collect(DISTINCT m) AS targets;
```

## Main Outputs
- `Layer10_Writeup.pdf`
- `memory_graph.json`
- `data/raw/airflow_issues.json`
- `data/processed/airflow_clean.json`
- `data/claims/claims_v1.json`
- `data/claims/claims_llm_v1.json`
- `data/claims/claims_dedup_v1.json`
- `data/Entities/entities_v1.json`
- `context_packs_examples/*.json`
- `Visualization/cytoscape_graph.html`

## Troubleshooting
- `ServiceUnavailable / DNS` from Neo4j:
  - check `NEO4J_URI` and network access
- Empty retrieval results:
  - rerun loaders in `Graph_Design/` and verify claims/evidence loaded
- Ollama errors:
  - ensure `ollama serve` is running and models are pulled
