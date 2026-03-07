# Retrieval and Grounding

Use this script to fetch grounded context packs from Neo4j:

`Retrieval/retrieval_grounding.py`

Streamlit UI:

`Retrieval/app.py`

## Run

```bash
./.venv/bin/python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode keyword
```

Hybrid mode (keyword + Ollama embeddings):

```bash
./.venv/bin/python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode hybrid \
  --embedding-model nomic-embed-text
```

Context pack only (skip LLM answer):

```bash
./.venv/bin/python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode keyword \
  --skip-answer \
  --format json \
  --pretty
```

Save output JSON:

```bash
./.venv/bin/python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode keyword \
  --format json \
  --output Retrieval/context_pack_example.json \
  --pretty
```

Print JSON to terminal:

```bash
./.venv/bin/python3 Retrieval/retrieval_grounding.py \
  --question "Who created issue 62799?" \
  --mode keyword \
  --format json \
  --pretty
```

## Streamlit UI

From `10/`:

```bash
./.venv/bin/streamlit run Retrieval/app.py
```

UI includes:
- question input
- retrieval mode toggle (`keyword` / `hybrid`)
- answer generation toggle
- limits for claims/evidence
- grounded claims/evidence display

## Controls for Explosion

- `--max-claims-per-entity` (default `10`)
- `--max-evidence-per-claim` (default `3`)
- `--max-claims-total` (default `30`)

## Notes

- Every returned claim is grounded: claims without evidence are excluded.
- Evidence includes source metadata and URL citations.
- Ambiguous questions return a follow-up suggestion with top issue candidates.
- Conflicting claims are surfaced in the `conflicts` section.
