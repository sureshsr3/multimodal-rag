"""
Contrastive fine-tuning of MultimodalEncoder projection heads.
Pulls cross-modal pairs into the same 512-d space using NT-Xent loss.

Usage:
    python -m training.train \
        --pairs_csv data/pairs.csv \
        --epochs 10 \
        --batch_size 32 \
        --lr 3e-4 \
        --output checkpoints/encoder.pt
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.encoders import MultimodalEncoder
from training.dataset import ContrastivePairDataset, collate_text_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on {device}")

    encoder = MultimodalEncoder(
        output_dim=512,
        freeze_backbones=args.freeze_backbones,
    ).to(device)

    if args.resume and Path(args.resume).exists():
        encoder.load_state_dict(torch.load(args.resume, map_location=device))
        log.info(f"Resumed from {args.resume}")

    dataset = ContrastivePairDataset(args.pairs_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_text_pairs,
        num_workers=0,
    )

    # Only train projection heads unless backbones are unfrozen
    params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        total_loss = 0.0

        for anchors, positives in loader:
            emb_a = encoder.encode_text(anchors)
            emb_p = encoder.encode_text(positives)
            loss = encoder.contrastive_loss(emb_a, emb_p)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        log.info(f"Epoch {epoch}/{args.epochs} — loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            encoder.save(args.output)
            log.info(f"Saved checkpoint → {args.output}")

    log.info("Training complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="checkpoints/encoder.pt")
    p.add_argument("--resume", default="")
    p.add_argument("--freeze_backbones", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
