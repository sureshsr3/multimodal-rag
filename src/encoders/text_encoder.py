"""
Text encoder: wraps a pretrained transformer and projects to shared embedding space.
Uses mean-pooling over token embeddings with an optional MLP projection head.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_dim),
        )

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        outputs = self.backbone(**encoded)
        pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return self.projection(pooled)
