"""
Sub-Sentence Encoder 학습 스크립트 (경량 버전).

사용 예:
    python train.py --config configs/base.yaml \
                     --data data/toy_propositions.jsonl \
                     --epochs 3 \
                     --output_dir checkpoints/toy_run
"""

from __future__ import annotations

import argparse
import functools
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from sub_sentence_encoder import (
    InfoNCELoss,
    PropositionPairDataset,
    SubSentenceEncoder,
    collate_fn,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data", type=str, required=True, help="학습용 JSONL 경로")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="config 값을 덮어쓰고 싶을 때")
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    set_seed(cfg.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    model = SubSentenceEncoder(
        encoder_name=cfg["encoder_name"],
        projection_size=cfg["projection_size"],
        dropout=cfg["dropout"],
    ).to(device)

    dataset = PropositionPairDataset(args.data)
    collate = functools.partial(
        collate_fn, tokenizer=tokenizer, max_seq_len=cfg["max_seq_len"]
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        drop_last=True if len(dataset) >= cfg["batch_size"] else False,
    )

    loss_fn = InfoNCELoss(temperature=cfg["temperature"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    total_steps = max(1, (len(loader) // cfg["grad_accum_steps"]) * cfg["epochs"])
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"] and device.type == "cuda")

    global_step = 0
    model.train()
    for epoch in range(cfg["epochs"]):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}")
        optimizer.zero_grad()
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=cfg["fp16"] and device.type == "cuda"):
                emb_a = model(
                    batch["input_ids_a"], batch["attention_mask_a"], batch["prop_mask_a"]
                )
                emb_b = model(
                    batch["input_ids_b"], batch["attention_mask_b"], batch["prop_mask_b"]
                )
                loss = loss_fn(emb_a, emb_b) / cfg["grad_accum_steps"]

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg["grad_accum_steps"]

            if (step + 1) % cfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    pbar.set_postfix(loss=running_loss / (step + 1))

        print(f"[epoch {epoch + 1}] avg_loss={running_loss / max(1, len(loader)):.4f}")

    save_path = os.path.join(args.output_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        save_path,
    )
    tokenizer.save_pretrained(args.output_dir)
    print(f"[info] saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
