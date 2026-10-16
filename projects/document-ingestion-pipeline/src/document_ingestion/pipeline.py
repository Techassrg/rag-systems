"""Validate text documents and split them into stable overlapping chunks."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass

SUPPORTED_EXTENSIONS = {".md", ".txt"}
DEFAULT_MAX_DOCUMENT_BYTES = 1_000_000


class IngestionError(ValueError):
    """Raised when a document violates the ingestion contract."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_name: str
    index: int
    text: str


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    source_name: str
    source_bytes: int
    chunks: tuple[Chunk, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "source_name": self.source_name,
            "source_bytes": self.source_bytes,
            "chunk_count": len(self.chunks),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }


def ingest_document(
    filename: str,
    content: bytes,
    *,
    chunk_size: int = 120,
    overlap: int = 20,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
) -> IngestionResult:
    """Ingest one UTF-8 Markdown/text document using word-count chunk bounds.

    IDs are stable for identical filename/content/chunking inputs. This module
    accepts bytes rather than paths so callers can apply their own storage policy.
    """
    if not filename or filename in {".", ".."} or "\x00" in filename:
        raise IngestionError("filename is required and must be a plain file name")
    if "/" in filename or "\\" in filename:
        raise IngestionError("paths are not accepted; provide a file name only")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if f".{suffix}" not in SUPPORTED_EXTENSIONS:
        raise IngestionError("only .md and .txt documents are supported")
    if not isinstance(content, bytes):
        raise IngestionError("content must be bytes")
    if max_document_bytes < 1 or len(content) > max_document_bytes:
        raise IngestionError("document exceeds the configured byte limit")
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise IngestionError("chunk_size must be positive and overlap smaller than chunk_size")

    try:
        decoded = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise IngestionError("document must be valid UTF-8 text") from exc

    normalized = unicodedata.normalize("NFC", decoded.replace("\r\n", "\n").replace("\r", "\n"))
    words = normalized.split()
    if not words:
        raise IngestionError("document contains no text")

    source_name = filename
    document_id = hashlib.sha256(source_name.encode("utf-8") + b"\0" + normalized.encode("utf-8")).hexdigest()[:20]
    step = chunk_size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        text = " ".join(chunk_words)
        chunk_id = hashlib.sha256(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()[:24]
        chunks.append(Chunk(chunk_id, document_id, source_name, index, text))

    return IngestionResult(document_id, source_name, len(content), tuple(chunks))
