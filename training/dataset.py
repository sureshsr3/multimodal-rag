"""
Contrastive training dataset: pairs of (anchor, positive) across modalities.
For multimodal alignment we need cross-modal pairs:
  - (audio_transcript, receipt_ocr_text)  — both describe the same event
  - (meeting_note_text, audio_transcript)
  - (receipt_image, ocr_text)

In practice, construct pairs from your domain data.
This file provides a generic base + a CSV-backed loader.
"""
from __future__ import annotations
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ContrastivePairDataset(Dataset):
    """
    CSV format: anchor_text, positive_text, anchor_modality, positive_modality
    """

    def __init__(self, csv_path: str | Path):
        self.pairs: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append(row)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]


class InMemoryPairDataset(Dataset):
    """For programmatic pair construction."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, positive = self.pairs[idx]
        return {"anchor_text": anchor, "positive_text": positive}


def collate_text_pairs(batch: list[dict]) -> tuple[list[str], list[str]]:
    anchors = [b["anchor_text"] for b in batch]
    positives = [b["positive_text"] for b in batch]
    return anchors, positives
