"""
Sub-Sentence Encoder core model.

논문 3.1절(Architecture)의 최소 구현:
  1. 입력 문장을 사전학습 인코더(예: BERT)에 한 번만 통과시킨다.
  2. 문장 내 각 명제(proposition)를 나타내는 binary token mask를 사용해
     해당 토큰들의 hidden state를 mean pooling한다.
  3. pooled vector를 작은 projection MLP에 통과시켜 최종 명제 임베딩을 얻는다.

핵심 포인트: 인코더는 문장 전체에 대해 "한 번만" forward하고, pooling만 마스크별로
반복하기 때문에 k개의 명제를 인코딩해도 추론 비용이 크게 늘지 않는다 (논문에서 강조하는 부분).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class ProjectionMLP(nn.Module):
    """Pooled 벡터를 최종 임베딩 공간으로 사영하는 2-layer MLP."""

    def __init__(self, hidden_size: int, projection_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubSentenceEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        projection_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = ProjectionMLP(hidden_size, projection_size, dropout)

    def encode_sentences(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """문장을 인코더에 통과시켜 토큰별 hidden state를 반환. [B, L, H]"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor, prop_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        hidden_states: [B, L, H]
        prop_mask:     [B, L]  (해당 명제에 속한 토큰만 1, 나머지는 0 — attention mask와는 별개)
        반환: [B, H]
        """
        prop_mask = prop_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (hidden_states * prop_mask).sum(dim=1)  # [B, H]
        counts = prop_mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prop_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        한 문장 안의 '하나의 명제'에 대한 임베딩을 계산.
        (여러 명제를 동시에 인코딩하려면 encode_sentences를 한 번 호출한 뒤
         masked_mean_pool을 여러 mask에 대해 반복 호출하면 된다. 학습 스크립트에서는
         positive pair 방식(문장 A의 명제, 문장 B의 명제)을 사용하므로 이 forward를
         두 번 호출하는 형태로 충분하다.)
        """
        hidden_states = self.encode_sentences(input_ids, attention_mask)
        pooled = self.masked_mean_pool(hidden_states, prop_mask)
        return self.projection(pooled)

    def encode_multi_proposition(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prop_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        한 문장에서 k개의 명제를 동시에 인코딩 (인코더는 1회만 forward).
        input_ids:      [B, L]
        attention_mask: [B, L]
        prop_masks:     [B, K, L]
        반환: [B, K, D]
        """
        hidden_states = self.encode_sentences(input_ids, attention_mask)  # [B, L, H]
        B, K, L = prop_masks.shape
        hidden_expanded = hidden_states.unsqueeze(1).expand(B, K, L, -1)  # [B, K, L, H]
        mask_expanded = prop_masks.unsqueeze(-1).float()  # [B, K, L, 1]
        summed = (hidden_expanded * mask_expanded).sum(dim=2)  # [B, K, H]
        counts = mask_expanded.sum(dim=2).clamp(min=1e-6)  # [B, K, 1]
        pooled = summed / counts  # [B, K, H]
        return self.projection(pooled)
