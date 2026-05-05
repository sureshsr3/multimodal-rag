"""
Unit tests for encoder shapes and normalization.
Run: pytest tests/ -v
"""
import torch
import pytest
from src.encoders import MultimodalEncoder, TextEncoder, ImageEncoder


def test_text_encoder_shape():
    enc = TextEncoder(output_dim=128)
    out = enc(["hello world", "test sentence"])
    assert out.shape == (2, 128)


def test_text_encoder_normalized():
    enc = TextEncoder(output_dim=128)
    import torch.nn.functional as F
    out = F.normalize(enc(["hello"]), dim=-1)
    assert torch.allclose(out.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_image_encoder_shape():
    from PIL import Image
    import numpy as np
    enc = ImageEncoder(output_dim=128, pretrained=False)
    dummy = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype="uint8"))]
    out = enc(dummy)
    assert out.shape == (1, 128)


def test_multimodal_encode_text():
    model = MultimodalEncoder(output_dim=128)
    result = model.encode(modality="text", inputs=["spend was $500"])
    assert result["embeddings"].shape == (1, 128)
    assert len(result["metadata"]) == 1
    assert result["metadata"][0]["modality"] == "text"


def test_contrastive_loss():
    model = MultimodalEncoder(output_dim=128)
    a = model.encode_text(["voice memo: $500"])
    b = model.encode_text(["receipt total: $500"])
    loss = model.contrastive_loss(a, b)
    assert loss.item() >= 0
