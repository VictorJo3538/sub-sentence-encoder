"""
논문 3.2절 학습 목적식의 최소 구현: in-batch negative를 사용하는 InfoNCE.

배치 안에서 (prop_a_i, prop_b_i)가 서로 의미적으로 동일한 positive pair이고,
그 외 모든 prop_b_j (j != i) 및 prop_a_j (j != i)는 negative로 취급한다.
(SimCSE 스타일 양방향 대조학습과 동일한 형태.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        emb_a, emb_b: [B, D], 이미 각각 명제 A / 명제 B의 임베딩.
        emb_a[i]와 emb_b[i]가 positive pair.
        """
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        # [B, B] 유사도 행렬: sim[i, j] = cos(a_i, b_j)
        sim_ab = torch.matmul(emb_a, emb_b.t()) / self.temperature
        labels = torch.arange(sim_ab.size(0), device=sim_ab.device)

        loss_ab = self.ce(sim_ab, labels)
        loss_ba = self.ce(sim_ab.t(), labels)  # 양방향 대칭 loss (b -> a 방향)

        return (loss_ab + loss_ba) / 2.0
