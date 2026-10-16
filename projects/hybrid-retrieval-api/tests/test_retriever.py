from fastapi.testclient import TestClient
import pytest

from hybrid_retrieval.app import create_app
from hybrid_retrieval.retriever import Document, HybridRetriever


def sample_retriever():
    return HybridRetriever(
        [
            Document("timeout", "payment.md", "payment adapter timeout left authorization pending"),
            Document("retry", "retry.md", "check transaction status and idempotency key before retrying payment"),
            Document("database", "database.md", "database connection pool reached its limit"),
        ]
    )


def test_hybrid_results_are_explainable_and_relevant():
    results = sample_retriever().search("payment timeout", top_k=2)

    assert [result["chunk_id"] for result in results][0] == "timeout"
    assert results[0]["score_breakdown"]["bm25_raw"] > 0
    assert results[0]["score_breakdown"]["tfidf_cosine_raw"] > 0


def test_empty_corpus_and_unmatched_query_return_no_results():
    assert HybridRetriever([]).search("payment") == []
    assert sample_retriever().search("unrelatedterm") == []


def test_rejects_duplicate_ids_and_invalid_top_k():
    with pytest.raises(ValueError, match="unique"):
        HybridRetriever([Document("same", "a", "first"), Document("same", "b", "second")])
    with pytest.raises(ValueError, match="top_k"):
        sample_retriever().search("payment", top_k=21)


def test_search_endpoint_contract_and_input_validation():
    client = TestClient(create_app(sample_retriever()))
    assert client.get("/health").json() == {"status": "ok"}

    response = client.post("/v1/search", json={"query": "payment timeout", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["results"][0]["chunk_id"] == "timeout"

    assert client.post("/v1/search", json={"query": "  "}).status_code == 422
    assert client.post("/v1/search", json={"query": "x", "top_k": 21}).status_code == 422
    assert client.post("/v1/search", json={"query": "x" * 501}).status_code == 422
