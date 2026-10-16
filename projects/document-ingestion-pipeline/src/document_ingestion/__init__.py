"""Safe, deterministic document ingestion primitives."""

from .pipeline import Chunk, IngestionError, IngestionResult, ingest_document

__all__ = ["Chunk", "IngestionError", "IngestionResult", "ingest_document"]
