"""
Integration-level tests for ingestion processors.
Uses only stdlib / minimal deps — no heavy ML models loaded.
"""
import tempfile
from pathlib import Path

from src.ingestion.text_processor import TextProcessor


def test_text_processor_chunks():
    proc = TextProcessor(chunk_size=10, chunk_overlap=2)
    chunks = proc.process("word " * 50, source="test.txt")
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.strip()
        assert c.source == "test.txt"


def test_text_processor_file():
    proc = TextProcessor()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document. " * 20)
        f.flush()
        chunks = proc.process_file(f.name)
    assert chunks
    assert all(c.modality == "text" for c in chunks)
