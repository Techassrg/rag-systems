import pytest

from document_ingestion import IngestionError, ingest_document


def test_normalizes_and_chunks_with_deterministic_ids():
    content = ("one two three four five six seven eight nine ten " * 2).encode()
    first = ingest_document("incident.md", content, chunk_size=6, overlap=2)
    second = ingest_document("incident.md", content, chunk_size=6, overlap=2)

    assert first == second
    assert [len(chunk.text.split()) for chunk in first.chunks] == [6, 6, 6, 6, 4]
    assert first.chunks[0].text.split()[-2:] == first.chunks[1].text.split()[:2]


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("private.pdf", b"text", "only .md and .txt"),
        ("../incident.txt", b"text", "file name only"),
        ("incident.txt", b"\xff", "valid UTF-8"),
        ("incident.txt", b"  \n", "contains no text"),
    ],
)
def test_rejects_unsupported_or_malformed_documents(filename, content, message):
    with pytest.raises(IngestionError, match=message):
        ingest_document(filename, content)


def test_enforces_size_and_chunk_configuration_limits():
    with pytest.raises(IngestionError, match="byte limit"):
        ingest_document("incident.txt", b"too large", max_document_bytes=4)
    with pytest.raises(IngestionError, match="overlap smaller"):
        ingest_document("incident.txt", b"some content", chunk_size=4, overlap=4)
