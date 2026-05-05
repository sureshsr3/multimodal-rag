"""
Splits plain text into overlapping chunks suitable for embedding.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    source: str
    modality: str = "text"
    metadata: dict = field(default_factory=dict)


class TextProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split(self, text: str) -> list[str]:
        # Split on sentence boundaries first, then merge into chunks
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks, current, current_len = [], [], 0

        for sentence in sentences:
            tokens = sentence.split()
            if current_len + len(tokens) > self.chunk_size and current:
                chunks.append(" ".join(current))
                # keep overlap
                overlap_tokens = current[-self.chunk_overlap:] if self.chunk_overlap else []
                current = overlap_tokens + tokens
                current_len = len(current)
            else:
                current.extend(tokens)
                current_len += len(tokens)

        if current:
            chunks.append(" ".join(current))

        return chunks

    def process(self, text: str, source: str = "unknown") -> list[TextChunk]:
        return [
            TextChunk(text=chunk, chunk_index=i, source=source)
            for i, chunk in enumerate(self._split(text))
        ]

    def process_file(self, path: str) -> list[TextChunk]:
        with open(path, "r", encoding="utf-8") as f:
            return self.process(f.read(), source=path)
