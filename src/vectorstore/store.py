"""
ChromaDB-backed vector store with per-modality collections.
All embeddings live in the same 512-d space from MultimodalEncoder,
stored together in one collection with a 'modality' metadata filter.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import uuid

import chromadb
from chromadb.config import Settings
import numpy as np


class MultimodalVectorStore:
    COLLECTION_NAME = "multimodal_rag"

    def __init__(self, persist_dir: str | Path | None = "./data/chroma_db"):
        """
        persist_dir=None  → ephemeral in-memory store (HF Spaces, testing)
        persist_dir=path  → persistent ChromaDB on disk (local / Docker)
        """
        if persist_dir is None:
            self.persist_dir = None
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.persist_dir = str(persist_dir)
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    #  Add                                                                 #
    # ------------------------------------------------------------------ #

    def add(
        self,
        embeddings: list[list[float]] | np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        # ChromaDB requires metadata values to be str/int/float/bool
        clean_meta = []
        for m in metadatas:
            clean_meta.append({
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in m.items()
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_meta,
        )
        return ids

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_embedding: list[float] | np.ndarray,
        n_results: int = 5,
        modality_filter: str | None = None,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        if modality_filter:
            where = where or {}
            where["modality"] = {"$eq": modality_filter}

        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine sim
            }
            for i in range(len(results["ids"][0]))
        ]

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def count(self, modality: str | None = None) -> int:
        if modality is None:
            return self.collection.count()
        results = self.collection.get(where={"modality": {"$eq": modality}})
        return len(results["ids"])

    def delete_by_source(self, source: str):
        results = self.collection.get(where={"source": {"$eq": source}})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def list_sources(self) -> list[str]:
        results = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in results["metadatas"]}
        return sorted(s for s in sources if s)

    def reset(self):
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
