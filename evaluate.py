"""
학습된 Sub-Sentence Encoder로 명제 쌍 검색(retrieval) 정확도를 확인하는 최소 평가 스크립트.

각 문장 A의 명제 임베딩에 대해, 전체 문장 B 명제 임베딩들 중 코사인 유사도가 가장 높은 것이
실제 정답(같은 줄의 B)인지를 확인하여 Recall@1을 계산한다.

사용 예:
    python evaluate.py --checkpoint checkpoints/toy_run --data data/toy_propositions.jsonl
"""

from __future__ import annotations

import argparse
import functools
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sub_sentence_encoder import PropositionPairDataset, SubSentenceEncoder, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="train.py의 output_dir")
    parser.add_argument("--data", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(os.path.join(args.checkpoint, "model.pt"), map_location=device)
    cfg = ckpt["config"]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = SubSentenceEncoder(
        encoder_name=cfg["encoder_name"],
        projection_size=cfg["projection_size"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = PropositionPairDataset(args.data)
    collate = functools.partial(
        collate_fn, tokenizer=tokenizer, max_seq_len=cfg["max_seq_len"]
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate)

    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        emb_a = model(batch["input_ids_a"], batch["attention_mask_a"], batch["prop_mask_a"])
        emb_b = model(batch["input_ids_b"], batch["attention_mask_b"], batch["prop_mask_b"])

        emb_a = torch.nn.functional.normalize(emb_a, dim=-1)
        emb_b = torch.nn.functional.normalize(emb_b, dim=-1)

        sim = torch.matmul(emb_a, emb_b.t())  # [N, N]
        preds = sim.argmax(dim=1)
        labels = torch.arange(sim.size(0), device=device)
        recall_at_1 = (preds == labels).float().mean().item()

    print(f"[eval] N={len(dataset)}  Recall@1 = {recall_at_1:.4f}")
    print("주의: toy 데이터(10개)는 파이프라인 동작 확인용이며, 정확도 수치 자체는 의미가 크지 않습니다.")


if __name__ == "__main__":
    main()
