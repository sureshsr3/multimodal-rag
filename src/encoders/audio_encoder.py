"""
Audio encoder: Whisper transcribes audio → transcript fed into TextEncoder.
Retains audio-specific metadata (duration, speaker segments if available).
"""
import os
import torch
import torch.nn as nn
from pathlib import Path


class AudioEncoder(nn.Module):
    def __init__(self, text_encoder: nn.Module, whisper_model: str = "base"):
        """
        Args:
            text_encoder: A TextEncoder instance for embedding transcripts.
            whisper_model: Whisper model size (tiny/base/small/medium/large).
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.whisper_model_name = whisper_model
        self._whisper = None  # lazy-loaded to avoid heavy import at startup

    def _load_whisper(self):
        if self._whisper is None:
            import whisper
            self._whisper = whisper.load_model(self.whisper_model_name)
        return self._whisper

    def transcribe(self, audio_path: str | Path) -> dict:
        """Return Whisper result dict with 'text' and 'segments'."""
        model = self._load_whisper()
        result = model.transcribe(str(audio_path), fp16=torch.cuda.is_available())
        return {
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", "en"),
            "source_path": str(audio_path),
        }

    def forward(self, audio_paths: list[str | Path]) -> tuple[torch.Tensor, list[dict]]:
        """
        Returns:
            embeddings: (N, output_dim) tensor
            transcripts: list of transcript dicts with text + segments
        """
        transcripts = [self.transcribe(p) for p in audio_paths]
        texts = [t["text"] for t in transcripts]
        embeddings = self.text_encoder(texts)
        return embeddings, transcripts
