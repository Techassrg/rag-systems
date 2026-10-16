"""Deterministic BM25 plus sparse TF-IDF cosine retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class Document:
    chunk_id: str
    source_name: str
    text: str


class HybridRetriever:
    """Small local retriever with separately exposed lexical and vector scores."""

    def __init__(self, documents: list[Document], *, lexical_weight: float = 0.55) -> None:
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical_weight must be between 0 and 1")
        ids = [document.chunk_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("chunk_id values must be unique")
        self.documents = documents
        self.lexical_weight = lexical_weight
        self.tokens = [tokenize(document.text) for document in documents]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.document_frequency: Counter[str] = Counter()
        for terms in self.term_counts:
            self.document_frequency.update(terms.keys())
        self.document_count = len(documents)

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, object]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.documents:
            return []
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        lexical = [self._bm25(query_tokens, index) for index in range(self.document_count)]
        vector = [self._cosine(query_tokens, index) for index in range(self.document_count)]
        lexical_max = max(lexical, default=0.0)
        vector_max = max(vector, default=0.0)
        vector_weight = 1.0 - self.lexical_weight
        ranked = []
        for index, document in enumerate(self.documents):
            lexical_normalized = lexical[index] / lexical_max if lexical_max else 0.0
            vector_normalized = vector[index] / vector_max if vector_max else 0.0
            combined = self.lexical_weight * lexical_normalized + vector_weight * vector_normalized
            ranked.append(
                {
                    "chunk_id": document.chunk_id,
                    "source_name": document.source_name,
                    "text": document.text,
                    "score": round(combined, 6),
                    "score_breakdown": {
                        "bm25_raw": round(lexical[index], 6),
                        "tfidf_cosine_raw": round(vector[index], 6),
                        "bm25_normalized": round(lexical_normalized, 6),
                        "tfidf_cosine_normalized": round(vector_normalized, 6),
                    },
                }
            )
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
        return [item for item in ranked[:top_k] if float(item["score"]) > 0]

    def _idf(self, term: str) -> float:
        df = self.document_frequency[term]
        return math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))

    def _bm25(self, query_tokens: list[str], index: int) -> float:
        if self.average_length == 0:
            return 0.0
        k1, b = 1.5, 0.75
        score = 0.0
        counts = self.term_counts[index]
        for term in sorted(set(query_tokens)):
            frequency = counts[term]
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * self.lengths[index] / self.average_length)
            score += self._idf(term) * frequency * (k1 + 1) / denominator
        return score

    def _tfidf_vector(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values())
        if not total:
            return {}
        return {term: (frequency / total) * self._idf(term) for term, frequency in counts.items()}

    def _cosine(self, query_tokens: list[str], index: int) -> float:
        query_vector = self._tfidf_vector(Counter(query_tokens))
        document_vector = self._tfidf_vector(self.term_counts[index])
        if not query_vector or not document_vector:
            return 0.0
        dot = sum(weight * document_vector.get(term, 0.0) for term, weight in query_vector.items())
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        document_norm = math.sqrt(sum(weight * weight for weight in document_vector.values()))
        return dot / (query_norm * document_norm) if query_norm and document_norm else 0.0
