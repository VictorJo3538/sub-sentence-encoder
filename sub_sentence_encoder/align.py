"""
논문 §3.3 "Step 3: Convert Propositions ⇒ Token Masks" / Appendix A.3의 재구현.

SegmenT5 같은 생성 모델이 만든 명제 텍스트는 원문 문장을 그대로 인용하지 않고
패러프레이징하는 경우가 많다 (예: 어순 변경, 대명사 치환 등). 그래서 단순 부분
문자열 탐색(str.find)으로는 span을 못 찾는 경우가 많다.

논문은 이 문제를, (1) 명제와 문장 각각을 NLTK로 토큰화·표제어화(lemmatize)하고,
(2) 표제어가 같은 토큰 쌍에 유사도 1을 부여한 affinity matrix를 만들고,
(3) 3-토큰 윈도우 내 다른 매치에는 작은 보너스 점수를 주는 2D 컨볼루션으로 동점을
    깨고, (4) Hungarian algorithm(최대 이분 매칭)으로 명제 토큰 <-> 문장 토큰의
    최적 대응을 찾는 방식으로 해결한다.

이 모듈은 그 파이프라인을 최소 재현한다. 매칭된 문장 토큰들의 character span을
모두 포함하는 [min_start, max_end) 구간을 최종 span으로 반환한다.
"""

from __future__ import annotations

import string

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from scipy.optimize import linear_sum_assignment

_lemmatizer = WordNetLemmatizer()
_tokenizer = TreebankWordTokenizer()


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    """(token, char_start, char_end) 목록을 반환."""
    spans = list(_tokenizer.span_tokenize(text))
    return [(text[s:e], s, e) for s, e in spans]


def _lemma(token: str) -> str:
    return _lemmatizer.lemmatize(token.lower())


def _is_punct(token: str) -> bool:
    return all(ch in string.punctuation for ch in token)


def _affinity_matrix(prop_tokens: list[str], sent_tokens: list[str]) -> np.ndarray:
    """prop_tokens[i] <-> sent_tokens[j] 표제어 일치 시 1.0, 아니면 0.0. shape [P, S]
    구두점 토큰(마침표 등)은 문장 어디서나 나타나 매칭을 왜곡시키므로 제외한다."""
    prop_lemmas = [_lemma(t) for t in prop_tokens]
    sent_lemmas = [_lemma(t) for t in sent_tokens]
    mat = np.zeros((len(prop_tokens), len(sent_tokens)), dtype=np.float64)
    for i, (pt, pl) in enumerate(zip(prop_tokens, prop_lemmas)):
        if _is_punct(pt):
            continue
        for j, (st, sl) in enumerate(zip(sent_tokens, sent_lemmas)):
            if _is_punct(st):
                continue
            if pl == sl:
                mat[i, j] = 1.0
    return mat


def _apply_window_bonus(mat: np.ndarray, window: int = 3, bonus: float = 0.1) -> np.ndarray:
    """3토큰 컨텍스트 윈도우 내 다른 위치의 매치에 작은 보너스를 더해 동점을 깨는
    2D 컨볼루션(논문 표현). 대각 방향으로 가까운 매치를 선호하게 만드는 효과가 있다."""
    P, S = mat.shape
    if P == 0 or S == 0:
        return mat
    out = mat.copy()
    offsets = range(-window, window + 1)
    for di in offsets:
        for dj in offsets:
            if di == 0 and dj == 0:
                continue
            shifted = np.zeros_like(mat)
            src_i0, src_i1 = max(0, -di), min(P, P - di)
            src_j0, src_j1 = max(0, -dj), min(S, S - dj)
            dst_i0, dst_i1 = max(0, di), min(P, P + di)
            dst_j0, dst_j1 = max(0, dj), min(S, S + dj)
            shifted[dst_i0:dst_i1, dst_j0:dst_j1] = mat[src_i0:src_i1, src_j0:src_j1]
            out += bonus * shifted
    return out


def align_proposition_to_span(sentence: str, proposition: str) -> list[int] | None:
    """
    proposition(패러프레이징된 명제 텍스트)을 sentence 안에서 가장 잘 대응하는
    character span [start, end)으로 정렬한다. 논문 Appendix A.3 방식.

    매칭되는 표제어 쌍이 하나도 없으면 None을 반환한다.
    """
    prop_tok_spans = _tokenize_with_spans(proposition)
    sent_tok_spans = _tokenize_with_spans(sentence)
    if not prop_tok_spans or not sent_tok_spans:
        return None

    prop_tokens = [t for t, _, _ in prop_tok_spans]
    sent_tokens = [t for t, _, _ in sent_tok_spans]

    affinity = _affinity_matrix(prop_tokens, sent_tokens)
    if affinity.sum() == 0:
        return None  # 표제어가 하나도 겹치지 않음 -> 정렬 불가

    scored = _apply_window_bonus(affinity)

    # Hungarian algorithm은 정사각 행렬을 기본으로 가정하므로, 비용 행렬로 변환하고
    # (-scored) 최소화 형태로 최대 매칭을 구한다. 두 토큰 수가 다르면 scipy가
    # min(P, S)개의 매칭만 반환한다 (직사각 입력 지원).
    row_idx, col_idx = linear_sum_assignment(-scored)

    matched_sent_idx = sorted(
        j for i, j in zip(row_idx, col_idx) if affinity[i, j] > 0  # 실제로 표제어가 일치한 매칭만 채택
    )
    if not matched_sent_idx:
        return None

    best_cluster = _largest_contiguous_cluster(matched_sent_idx, max_gap=3)

    starts = [sent_tok_spans[j][1] for j in best_cluster]
    ends = [sent_tok_spans[j][2] for j in best_cluster]
    return [min(starts), max(ends)]


def _largest_contiguous_cluster(indices: list[int], max_gap: int) -> list[int]:
    """정렬된 문장 토큰 인덱스들을, 인접 간격이 max_gap 이하인 그룹으로 묶은 뒤
    가장 원소가 많은 그룹을 반환한다. 전역 최적 매칭이 문장 곳곳에 흩어진 동음이의
    토큰(예: 흔한 단어)을 잘못 끌어오는 것을 걸러내는 역할을 한다."""
    if not indices:
        return []
    clusters: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx - clusters[-1][-1] <= max_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return max(clusters, key=len)
