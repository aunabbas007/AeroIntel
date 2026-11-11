"""Hybrid dense + BM25 retrieval with optional cross-encoder reranking.

The dense index remains the source of truth. BM25 is built from the same
versioned local data directory included in the application image. Results
are fused with Reciprocal Rank Fusion (RRF), then optionally reranked.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from source.chunking import split_documents
from source.loader import load_documents

logger = logging.getLogger("aerointel.retrieval")

RRF_CONSTANT = 60
DEFAULT_CANDIDATE_K = 12


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=2)
def _sparse_index(data_path: str):
    documents = split_documents(load_documents(data_path))
    if not documents:
        raise RuntimeError(f"No retrieval documents found under {data_path!r}.")
    corpus_tokens = [tokenize(doc.page_content) for doc in documents]
    return documents, BM25Okapi(corpus_tokens)


def reciprocal_rank_fusion(rankings: list[list], k: int = RRF_CONSTANT):
    """Merge ranked document lists without relying on incomparable scores."""
    scores = {}
    documents = {}

    for ranking in rankings:
        seen_in_ranking = set()
        for rank, document in enumerate(ranking, start=1):
            metadata = document.metadata or {}
            identity = (
                str(metadata.get("source", "")),
                document.page_content,
            )
            if identity in seen_in_ranking:
                continue
            seen_in_ranking.add(identity)
            documents[identity] = document
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)

    return [
        documents[identity]
        for identity in sorted(scores, key=scores.get, reverse=True)
    ]


def _rerank(query: str, documents: list, top_k: int) -> list:
    if os.getenv("AEROINTEL_RERANKER_ENABLED", "false").lower() != "true":
        return documents[:top_k]

    try:
        from sentence_transformers import CrossEncoder

        model_name = os.getenv(
            "AEROINTEL_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        model = _get_cross_encoder(model_name)
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)
        return [
            doc
            for _, doc in sorted(
                zip(scores, documents),
                key=lambda item: float(item[0]),
                reverse=True,
            )[:top_k]
        ]
    except Exception:
        # Reranking is an optional quality layer. Retrieval must remain
        # available if the model cannot be downloaded or initialized.
        logger.exception("reranker_unavailable_using_rrf_order")
        return documents[:top_k]


@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def hybrid_retrieve(
    query: str,
    vector_store,
    k: int = 3,
    entity_type: str | None = None,
    data_path: str = "data",
    candidate_k: int = DEFAULT_CANDIDATE_K,
):
    """Retrieve dense and sparse candidates, fuse with RRF, return top-k."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if k < 1 or candidate_k < k:
        raise ValueError("candidate_k must be greater than or equal to k")
    if entity_type not in {None, "airline", "aircraft"}:
        raise ValueError("entity_type must be 'airline', 'aircraft', or None")

    search_kwargs = {"k": candidate_k}
    if entity_type:
        search_kwargs["filter"] = {"type": entity_type}

    dense_documents = vector_store.similarity_search(query, **search_kwargs)

    sparse_documents, bm25 = _sparse_index(str(Path(data_path)))
    sparse_scores = bm25.get_scores(tokenize(query))
    sparse_order = sorted(
        range(len(sparse_documents)),
        key=lambda index: float(sparse_scores[index]),
        reverse=True,
    )

    if entity_type:
        sparse_order = [
            index for index in sparse_order
            if sparse_documents[index].metadata.get("type") == entity_type
        ]

    sparse_results = [
        sparse_documents[index]
        for index in sparse_order[:candidate_k]
        if sparse_scores[index] > 0
    ]

    fused = reciprocal_rank_fusion([dense_documents, sparse_results])
    return _rerank(query, fused, k)
