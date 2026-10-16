"""FastAPI wrapper around the in-memory hybrid retriever."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .retriever import Document, HybridRetriever


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, object]]


def load_fixture_retriever(path: Path | None = None) -> HybridRetriever:
    fixture_path = path or Path(__file__).resolve().parents[2] / "fixtures" / "chunks.jsonl"
    documents = []
    for line in fixture_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        documents.append(Document(chunk_id=item["chunk_id"], source_name=item["source_name"], text=item["text"]))
    return HybridRetriever(documents)


def create_app(retriever: HybridRetriever | None = None) -> FastAPI:
    app = FastAPI(title="Hybrid Retrieval API", version="0.1.0")
    active_retriever = retriever if retriever is not None else load_fixture_retriever()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must contain non-whitespace characters")
        return SearchResponse(query=request.query, results=active_retriever.search(request.query, top_k=request.top_k))

    return app


app = create_app()
