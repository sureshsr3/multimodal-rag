"""
MultimodalEncoder: unified entry point that routes inputs to per-modality encoders,
L2-normalizes all outputs, and returns embeddings in a shared 512-d vector space.

Alignment strategy: contrastive pre-training (see training/train.py) pulls paired
cross-modal representations together using NT-Xent loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .audio_encoder import AudioEncoder


class MultimodalEncoder(nn.Module):
    MODALITIES = ("text", "image", "audio")

    def __init__(
        self,
        output_dim: int = 512,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model: str = "vit_base_patch16_224",
        whisper_model: str = "base",
        freeze_backbones: bool = False,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.temperature = temperature

        self.text_encoder = TextEncoder(
            model_name=text_model,
            output_dim=output_dim,
            freeze_backbone=freeze_backbones,
        )
        self.image_encoder = ImageEncoder(
            model_name=image_model,
            output_dim=output_dim,
            freeze_backbone=freeze_backbones,
        )
        self.audio_encoder = AudioEncoder(
            text_encoder=self.text_encoder,
            whisper_model=whisper_model,
        )

    # ------------------------------------------------------------------ #
    #  Per-modality encode                                                 #
    # ------------------------------------------------------------------ #

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        emb = self.text_encoder(texts)
        return F.normalize(emb, dim=-1)

    def encode_image(self, images: list) -> torch.Tensor:
        emb = self.image_encoder(images)
        return F.normalize(emb, dim=-1)

    def encode_audio(self, audio_paths: list[str | Path]) -> tuple[torch.Tensor, list[dict]]:
        emb, transcripts = self.audio_encoder(audio_paths)
        return F.normalize(emb, dim=-1), transcripts

    # ------------------------------------------------------------------ #
    #  Unified encode                                                      #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        modality: str,
        inputs: list,
    ) -> dict:
        """
        Returns dict with keys:
            embeddings: (N, D) normalized tensor
            metadata:   list of per-item dicts (transcript, source, etc.)
        """
        if modality not in self.MODALITIES:
            raise ValueError(f"modality must be one of {self.MODALITIES}")

        metadata = [{"modality": modality} for _ in inputs]

        if modality == "text":
            emb = self.encode_text(inputs)
        elif modality == "image":
            emb = self.encode_image(inputs)
        else:  # audio
            emb, transcripts = self.encode_audio(inputs)
            for i, t in enumerate(transcripts):
                metadata[i].update(t)

        return {"embeddings": emb, "metadata": metadata}

    # ------------------------------------------------------------------ #
    #  NT-Xent contrastive loss (used during training)                    #
    # ------------------------------------------------------------------ #

    def contrastive_loss(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Symmetric NT-Xent between two batches of normalized embeddings."""
        logits = torch.matmul(emb_a, emb_b.T) / self.temperature
        labels = torch.arange(len(emb_a), device=emb_a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "MultimodalEncoder":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
