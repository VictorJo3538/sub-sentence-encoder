"""
데이터 형식 (JSONL, 한 줄에 하나의 positive pair):

{
  "text_a": "Dracula is a novel by Bram Stoker featuring Count Dracula as the protagonist.",
  "span_a": [0, 34],
  "text_b": "The novel Dracula was written by Bram Stoker.",
  "span_b": [0, 46]
}

- text_a / text_b: 명제가 포함된 전체 문장 (문맥 유지를 위해 문장 전체를 그대로 둔다).
- span_a / span_b: 그 문장 안에서 "이 명제에 해당하는" 부분의 character 단위 [start, end).

collate_fn에서 fast tokenizer의 offset_mapping을 이용해 span과 겹치는 토큰들을
1로 표시하는 prop_mask를 자동 생성한다. (원 논문에서는 이 마스크를 SegmenT5의 출력이나
사람이 annotation한 PropSegmEnt 라벨로부터 얻는다. 이 구현에서는 데이터 자체에 span을
미리 넣어두는 방식으로 단순화했다.)
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Dataset


class PropositionPairDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.examples: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def _build_prop_mask(offset_mapping: list[tuple[int, int]], span: list[int]) -> list[int]:
    """offset_mapping(토큰별 char 범위)과 span(char 범위)이 겹치는 토큰을 1로 표시."""
    start, end = span
    mask = []
    for tok_start, tok_end in offset_mapping:
        if tok_start == tok_end:  # special token ([CLS], [SEP], padding)
            mask.append(0)
            continue
        overlap = not (tok_end <= start or tok_start >= end)
        mask.append(1 if overlap else 0)
    # 명제 span에 해당하는 토큰이 하나도 안 걸리는 경우를 대비한 안전장치:
    # 전부 0이면 최소 1개 토큰(가장 가까운 토큰)이라도 살려서 division-by-zero를 방지.
    if sum(mask) == 0 and len(mask) > 0:
        mask[0] = 1
    return mask


def collate_fn(batch: list[dict[str, Any]], tokenizer, max_seq_len: int = 128):
    texts_a = [ex["text_a"] for ex in batch]
    texts_b = [ex["text_b"] for ex in batch]
    spans_a = [ex["span_a"] for ex in batch]
    spans_b = [ex["span_b"] for ex in batch]

    enc_a = tokenizer(
        texts_a,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    enc_b = tokenizer(
        texts_b,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    prop_mask_a = torch.tensor(
        [
            _build_prop_mask(offsets.tolist(), span)
            for offsets, span in zip(enc_a["offset_mapping"], spans_a)
        ],
        dtype=torch.long,
    )
    prop_mask_b = torch.tensor(
        [
            _build_prop_mask(offsets.tolist(), span)
            for offsets, span in zip(enc_b["offset_mapping"], spans_b)
        ],
        dtype=torch.long,
    )

    return {
        "input_ids_a": enc_a["input_ids"],
        "attention_mask_a": enc_a["attention_mask"],
        "prop_mask_a": prop_mask_a,
        "input_ids_b": enc_b["input_ids"],
        "attention_mask_b": enc_b["attention_mask"],
        "prop_mask_b": prop_mask_b,
    }
