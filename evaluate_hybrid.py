"""Compare AeroIntel dense retrieval against hybrid BM25 + dense retrieval.

Uses the curated test cases already maintained in evaluate_retreival.py.
This script measures retrieval only; evaluate_answers.py separately measures
end-to-end API answer behaviour.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from evaluate_retreival import TEST_CASES
from source.hybrid_search import hybrid_retrieve
from source.rag_metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from source.vector_store import create_vector_store

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "aerointel-v2")
K_VALUES = [3, 5, 8]


def _entity_rankings(documents):
    return [
        str(document.metadata.get("entity_id", ""))
        for document in documents
    ]


def _evaluate_mode(vector_store, mode: str, k: int):
    cases = [case for case in TEST_CASES if case["answerable"]]
    precisions = []
    recalls = []
    reciprocal_ranks = []
    keyword_coverages = []

    for case in cases:
        entity_type = case["type"]
        if mode == "dense":
            kwargs = {"k": k}
            if entity_type:
                kwargs["filter"] = {"type": entity_type}
            documents = vector_store.similarity_search(
                case["question"], **kwargs
            )
        elif mode == "hybrid":
            documents = hybrid_retrieve(
                query=case["question"],
                vector_store=vector_store,
                k=k,
                entity_type=entity_type,
            )
        else:
            raise ValueError(f"Unsupported evaluation mode: {mode}")

        ranking = _entity_rankings(documents)
        relevant = set(case["relevant_entities"])

        precisions.append(precision_at_k(ranking, relevant, k))
        recalls.append(recall_at_k(ranking, relevant, k))

        # One ranked result list per question; relevant entity IDs are the
        # ground-truth labels already maintained in the existing test set.
        reciprocal_ranks.append(
            next(
                (1.0 / rank for rank, entity in enumerate(ranking, 1)
                 if entity in relevant),
                0.0,
            )
        )

        retrieved_text = "\n".join(
            document.page_content for document in documents
        ).lower()
        terms = case["expected_terms"]
        if terms:
            keyword_coverages.append(
                sum(term.lower() in retrieved_text for term in terms)
                / len(terms)
            )

    count = len(cases)
    return {
        "precision": sum(precisions) / count if count else 0.0,
        "recall": sum(recalls) / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
        "keyword_coverage": (
            sum(keyword_coverages) / len(keyword_coverages)
            if keyword_coverages else 0.0
        ),
    }


def main():
    vector_store = create_vector_store(INDEX_NAME)
    print("AEROINTEL DENSE VS HYBRID RETRIEVAL EVALUATION")
    print(f"Index: {INDEX_NAME}")
    print(f"Answerable cases: {sum(case['answerable'] for case in TEST_CASES)}")

    for k in K_VALUES:
        for mode in ("dense", "hybrid"):
            metrics = _evaluate_mode(vector_store, mode, k)
            print(
                f"k={k:<2} mode={mode:<6} "
                f"Precision@K={metrics['precision']:.3f} "
                f"Recall@K={metrics['recall']:.3f} "
                f"MRR={metrics['mrr']:.3f} "
                f"KeywordCoverage={metrics['keyword_coverage']:.3f}"
            )

    print("\nChoose hybrid for production only if its measured quality is better")
    print("and latency/model-loading cost remain acceptable.")


if __name__ == "__main__":
    main()
