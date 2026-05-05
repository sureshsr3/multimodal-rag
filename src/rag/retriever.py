"""
Retriever: encodes a user query (text) and fetches relevant chunks
from the vector store, optionally filtered by modality.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from src.encoders import MultimodalEncoder
from src.vectorstore import MultimodalVectorStore


@dataclass
class RetrievedChunk:
    document: str
    metadata: dict
    score: float
    modality: str
    source: str


class MultimodalRetriever:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        store: MultimodalVectorStore,
        top_k: int = 5,
    ):
        self.encoder = encoder
        self.store = store
        self.top_k = top_k

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        modality_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        self.encoder.eval()

        query_emb = self.encoder.encode_text([query])[0]
        raw = self.store.query(
            query_embedding=query_emb.cpu().numpy(),
            n_results=k,
            modality_filter=modality_filter,
        )

        return [
            RetrievedChunk(
                document=r["document"],
                metadata=r["metadata"],
                score=r["score"],
                modality=r["metadata"].get("modality", "unknown"),
                source=r["metadata"].get("source", "unknown"),
            )
            for r in raw
        ]

    def retrieve_cross_modal(
        self,
        query: str,
        modalities: list[str] | None = None,
        top_k_per_modality: int = 3,
    ) -> dict[str, list[RetrievedChunk]]:
        """Retrieve top-k chunks per modality separately for comparison queries."""
        target_modalities = modalities or ["text", "image", "audio"]
        results = {}
        for mod in target_modalities:
            chunks = self.retrieve(query, top_k=top_k_per_modality, modality_filter=mod)
            if chunks:
                results[mod] = chunks
        return results
