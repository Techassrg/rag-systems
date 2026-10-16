"""Command line interface for deterministic local document ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import IngestionError, ingest_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and chunk a UTF-8 .md or .txt file")
    parser.add_argument("document", type=Path)
    parser.add_argument("--chunk-size", type=int, default=120, help="maximum words per chunk")
    parser.add_argument("--overlap", type=int, default=20, help="words shared by adjacent chunks")
    args = parser.parse_args()
    try:
        result = ingest_document(args.document.name, args.document.read_bytes(), chunk_size=args.chunk_size, overlap=args.overlap)
    except (OSError, IngestionError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
