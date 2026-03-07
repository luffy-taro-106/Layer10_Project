#!/usr/bin/env python3
"""Lightweight Streamlit UI for retrieval + grounding."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st #type: ignore

from retrieval_grounding import RetrievalGroundingService


def _run_retrieval(
    question: str,
    mode: str,
    llm_model: str,
    embedding_model: str,
    max_claims_per_entity: int,
    max_evidence_per_claim: int,
    max_claims_total: int,
    candidate_entity_limit: int,
    generate_answer: bool,
) -> Dict[str, Any]:
    service = RetrievalGroundingService(
        mode=mode,
        llm_model=llm_model,
        embedding_model=embedding_model,
        max_claims_per_entity=max_claims_per_entity,
        max_evidence_per_claim=max_evidence_per_claim,
        max_claims_total=max_claims_total,
        candidate_entity_limit=candidate_entity_limit,
    )
    try:
        pack = service.retrieve_context_pack(question)
        if generate_answer and not pack.get("error"):
            pack["answer"] = service.generate_answer(pack)
        return pack
    except Exception as e:
        return {
            "question": question,
            "retrieval_mode": mode,
            "error": {"type": type(e).__name__, "message": str(e)},
        }
    finally:
        service.close()


def _render_answer(pack: Dict[str, Any]) -> None:
    st.subheader("Answer")
    answer = pack.get("answer")
    if isinstance(answer, dict):
        text = str(answer.get("answer", "")).strip()
        if text:
            st.markdown(text)
        else:
            st.info("No answer text returned.")
    else:
        st.info("Answer generation skipped.")


def _render_claims(pack: Dict[str, Any]) -> None:
    claims: List[Dict[str, Any]] = pack.get("claims") or []
    st.subheader("Claims")
    if not claims:
        st.info("No grounded claims returned.")
        return
    rows = []
    for c in claims:
        rows.append(
            {
                "predicate": c.get("predicate"),
                "subject": c.get("subject"),
                "object": c.get("object"),
                "confidence": c.get("confidence"),
                "score": c.get("score"),
                "citations": ", ".join(c.get("citations") or []),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_evidence(pack: Dict[str, Any]) -> None:
    evidence: List[Dict[str, Any]] = pack.get("evidence") or []
    st.subheader("Evidence")
    if not evidence:
        st.info("No evidence returned.")
        return

    for ev in evidence:
        cid = ev.get("citation_id")
        source_type = ev.get("source_type")
        source_id = ev.get("source_id")
        excerpt = ev.get("excerpt") or ""
        url = ev.get("url") or ""

        st.markdown(f"**[{cid}]** `{source_type}` (`{source_id}`)")
        if excerpt:
            st.write(f"\"{excerpt}\"")
        if url:
            st.markdown(f"[Open source]({url})")
        st.divider()


def main() -> None:
    st.set_page_config(page_title="Retrieval + Grounding", layout="wide")
    st.title("Retrieval + Grounding")
    st.caption("Keyword / Hybrid retrieval with grounded claims and evidence citations.")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Retrieval mode", ["keyword", "hybrid"], index=0, horizontal=True)
        generate_answer = st.checkbox("Generate LLM answer", value=True)
        llm_model = st.text_input("LLM model", value="llama3")
        embedding_model = st.text_input(
            "Embedding model (hybrid)",
            value="nomic-embed-text",
            disabled=(mode != "hybrid"),
        )
        max_claims_per_entity = st.slider("Max claims per entity", 1, 20, 10)
        max_evidence_per_claim = st.slider("Max evidence per claim", 1, 5, 3)
        max_claims_total = st.slider("Max total claims", 1, 50, 30)
        candidate_entity_limit = st.slider("Candidate entity limit", 5, 50, 25)

    question = st.text_input("Question", value="Who created issue 62799?")
    run = st.button("Run Retrieval", type="primary")

    if run:
        with st.spinner("Retrieving grounded context..."):
            pack = _run_retrieval(
                question=question,
                mode=mode,
                llm_model=llm_model,
                embedding_model=embedding_model,
                max_claims_per_entity=max_claims_per_entity,
                max_evidence_per_claim=max_evidence_per_claim,
                max_claims_total=max_claims_total,
                candidate_entity_limit=candidate_entity_limit,
                generate_answer=generate_answer,
            )
        st.session_state["context_pack"] = pack

    pack = st.session_state.get("context_pack")
    if not pack:
        st.info("Enter a question and click Run Retrieval.")
        return

    if pack.get("error"):
        err = pack["error"]
        st.error(f"{err.get('type')}: {err.get('message')}")
        return

    ambiguity = pack.get("ambiguity") or {}
    if ambiguity.get("is_ambiguous"):
        st.warning(ambiguity.get("follow_up"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Entities", len(pack.get("entities") or []))
    with c2:
        st.metric("Claims", len(pack.get("claims") or []))
    with c3:
        st.metric("Evidence", len(pack.get("evidence") or []))

    _render_answer(pack)

    left, right = st.columns(2)
    with left:
        _render_claims(pack)
    with right:
        _render_evidence(pack)

    diagnostics = pack.get("diagnostics")
    if diagnostics:
        with st.expander("Diagnostics"):
            st.json(diagnostics)

    with st.expander("Raw Context Pack"):
        st.json(pack)


if __name__ == "__main__":
    main()
