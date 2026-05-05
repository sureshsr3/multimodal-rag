"""
Generator: local LLM via Ollama with streaming support.
Streaming means tokens appear in the UI as they are generated —
no waiting for the full response, no timeouts.
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests

from .retriever import RetrievedChunk


SYSTEM_PROMPT = """You are a question-answering assistant. You only answer from the CONTEXT provided.

RULES:
1. Use ONLY the information in the CONTEXT. Never add facts from your training.
2. For IMAGE DESCRIPTION context: report exactly what is described — objects, colors, setting.
3. For TEXT FROM IMAGE context: use only the extracted text.
4. For AUDIO TRANSCRIPT context: use only what was said in the recording.
5. If context is missing details the user asks about, say "The uploaded content does not mention that."
6. Never guess, invent, or expand beyond what the context says.
7. Write naturally — do not mention "context", "caption", "chunk", or "AI" in your answer.
8. End your answer with: Sources: <filename(s)>"""


@dataclass
class GeneratorResponse:
    answer: str
    sources: list[str]
    usage: dict
    retrieved_chunks: list[RetrievedChunk]


class RAGGenerator:
    def __init__(
        self,
        model: str | None = None,
        ollama_url: str | None = None,
        max_tokens: int = 512,          # keep short for CPU speed
    ):
        self.model      = model      or os.environ.get("OLLAMA_MODEL", "tinyllama")
        self.ollama_url = (ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.max_tokens = max_tokens

    def _is_ollama_running(self) -> bool:
        try:
            return requests.get(f"{self.ollama_url}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        sections = []
        for i, chunk in enumerate(chunks, 1):
            # Get chunk_type from metadata if available (caption vs ocr)
            chunk_type = chunk.metadata.get("chunk_type", "")
            modality   = chunk.modality

            if modality == "audio":
                label = "AUDIO RECORDING TRANSCRIPT"
            elif modality == "image" and chunk_type == "caption":
                label = "IMAGE DESCRIPTION"
            elif modality == "image" and chunk_type == "ocr":
                label = "TEXT FROM IMAGE"
            elif modality == "image":
                label = "IMAGE"
            else:
                label = "DOCUMENT"

            source_name = Path(chunk.source).name
            sections.append(
                f"[{i}] {label} — {source_name}\n"
                f"{chunk.document}"
            )
        return "\n\n---\n\n".join(sections)

    def _build_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict] | None = None,
    ) -> str:
        """
        history = [{"role": "user"|"assistant", "content": "..."},  ...]
        Previous turns are injected so the model can correct itself.
        """
        context_block = self._build_context(chunks)

        # Format previous conversation turns
        history_block = ""
        if history:
            turns = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                turns.append(f"{role}: {msg['content']}")
            history_block = "\n\nPrevious conversation:\n" + "\n".join(turns)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n\n{context_block}"
            f"{history_block}\n\n"
            f"---\n\nUser: {query}\n\nAssistant:"
        )

    # ------------------------------------------------------------------ #
    #  Streaming — yields one token string at a time                      #
    # ------------------------------------------------------------------ #
    def stream_tokens(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """
        Yields token strings as Ollama produces them.
        The FastAPI endpoint wraps this in a StreamingResponse.
        """
        if not self._is_ollama_running():
            raise RuntimeError(
                f"Ollama is not running at {self.ollama_url}. "
                "Open a terminal and run:  ollama serve"
            )

        prompt = self._build_prompt(query, retrieved_chunks, history)

        with requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model":  self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": 0.2,
                },
            },
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    # ------------------------------------------------------------------ #
    #  Non-streaming fallback (used by tests)                             #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> GeneratorResponse:
        full_answer = "".join(self.stream_tokens(query, retrieved_chunks))
        sources = list({c.source for c in retrieved_chunks})
        return GeneratorResponse(
            answer=full_answer.strip(),
            sources=sources,
            usage={"model": self.model, "prompt_tokens": 0, "response_tokens": 0, "total_duration_ms": 0},
            retrieved_chunks=retrieved_chunks,
        )
