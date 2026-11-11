from langchain_core.documents import Document
import pytest

from source.rag_metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from source.hybrid_search import (
    hybrid_retrieve,
    reciprocal_rank_fusion,
    tokenize,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 2) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(
        [["x", "a"], ["b", "x"]],
        [{"a"}, {"b"}],
    ) == pytest.approx(0.75)


def test_rrf_promotes_documents_ranked_by_both_retrievers():
    shared = Document(page_content="shared", metadata={"source": "shared.txt"})
    dense_only = Document(page_content="dense", metadata={"source": "dense.txt"})
    sparse_only = Document(page_content="sparse", metadata={"source": "sparse.txt"})

    fused = reciprocal_rank_fusion(
        [[shared, dense_only], [shared, sparse_only]]
    )

    assert fused[0].page_content == "shared"
    assert {doc.page_content for doc in fused} == {"shared", "dense", "sparse"}


def test_tokenize_normalizes_punctuation():
    assert tokenize("A350-1000: Trent XWB-97!") == [
        "a350", "1000", "trent", "xwb", "97"
    ]


def test_hybrid_retrieve_combines_dense_and_sparse_candidates(monkeypatch):
    import source.hybrid_search as module

    dense = Document(
        page_content="dense candidate",
        metadata={"source": "dense.txt", "type": "aircraft"},
    )
    sparse = Document(
        page_content="A350-1000 aircraft range",
        metadata={"source": "sparse.txt", "type": "aircraft"},
    )

    class FakeBM25:
        def get_scores(self, tokens):
            return [2.0]

    class FakeStore:
        def similarity_search(self, query, **kwargs):
            assert kwargs["filter"] == {"type": "aircraft"}
            return [dense]

    monkeypatch.setattr(
        module, "_sparse_index", lambda path: ([sparse], FakeBM25())
    )
    monkeypatch.setattr(
        module, "_rerank", lambda query, docs, top_k: docs[:top_k]
    )

    result = hybrid_retrieve(
        "A350-1000 range",
        FakeStore(),
        k=2,
        entity_type="aircraft",
    )

    assert {doc.page_content for doc in result} == {
        "dense candidate", "A350-1000 aircraft range"
    }


def test_hybrid_rejects_invalid_k():
    with pytest.raises(ValueError):
        hybrid_retrieve("aircraft", object(), k=4, candidate_k=3)


def test_invalid_metric_k_is_rejected():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)
