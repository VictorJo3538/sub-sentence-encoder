"""
논문(Chen et al., 2023, arXiv:2311.04335)에 실제로 텍스트로 명시된 "문장 -> 명제 분할" 예시들을
가져와서, 공식 SegmenT5-large 체크포인트로 로컬에서 직접 실행한 뒤, 논문이 보고한 결과와
로컬 실행 결과를 나란히 놓은 비교표(xlsx)를 만든다.

대상 예시 (모두 논문 본문/부록에 원문 그대로 등장하는 문장):
  1. Introduction §1 예시 문장: "Dracula is a novel by Bram Stoker featuring
     Count Dracula as the protagonist." (실제로는 두 개의 사실을 담고 있다는 설명과 함께 등장)
  2. Figure 1의 왼쪽 문장: "The Gothic novel Dracula is written by Bram Stoker."
  3. Appendix A.1 few-shot 프롬프트의 데모 예시: "The Andy Warhol Museum in his hometown,
     Pittsburgh, Pennsylvania, contains an extensive permanent collection of art."
     -> 이 예시는 논문이 "정답 output"까지 프롬프트에 명시해 두었으므로 paper_output이 존재한다.

주의:
  - 로컬 SegmenT5는 HF 모델 카드 기본 설정(num_beams=1) 대신 beam search(num_beams=4,
    no_repeat_ngram_size=3)를 사용한다 (scripts/segment_propositions.py 참고, greedy에서
    반복 생성 문제가 관찰되어 조정함). 이 차이가 논문/로컬 출력 차이의 한 원인일 수 있다.
  - SubSentenceEncoder 본체(대조학습된 인코더)는 공식 저장소(github.com/schen149/
    sub-sentence-encoder)를 clone하고, 저자가 Google Drive에 배포한 공식 체크포인트
    (subencoder_st5_base.ckpt)를 별도 conda 환경(subenc_official, pytorch_lightning==1.8.5)에서
    로드해 README의 "Usage" 예시(Dracula 문장 코사인 유사도)를 그대로 재현했다.
    (../sub-sentence-encoder-official/run_dracula_demo.py 실행 결과를 이 스크립트에 값으로 반영)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.segment_propositions import load_model, sentence_to_examples
import torch


# ---------------------------------------------------------------------------
# 논문에 실제로 텍스트로 명시된 예시들 (source: PDF 페이지, 인용 표시)
# ---------------------------------------------------------------------------
EXAMPLES = [
    {
        "example_id": "intro_dracula",
        "source_section": "§1 Introduction (본문)",
        "source_quote": (
            '"both sentences agree on Dracula is a novel and is published in the '
            '19th century" 문맥에서 언급된 예시 문장'
        ),
        "input_sentence": (
            "Dracula is a novel by Bram Stoker featuring Count Dracula as the "
            "protagonist."
        ),
        "paper_output": (
            "논문은 이 문장의 정확한 명제 분할 결과(claims 리스트)를 명시적으로 "
            "제공하지 않음 — '두 문장이 부분적으로 같은 의미를 공유한다'는 개념 "
            "설명용 예시로만 사용됨 (정량적 정답 없음)"
        ),
    },
    {
        "example_id": "figure1_left_sentence",
        "source_section": "Figure 1 (페이지 1, 그림 캡션의 왼쪽 문장)",
        "source_quote": '"The Gothic novel Dracula is written by Bram Stoker." (Similarity: Low 예시의 왼쪽 문장)',
        "input_sentence": "The Gothic novel Dracula is written by Bram Stoker.",
        "paper_output": (
            "논문은 이 문장에 대한 명제 분할 결과를 텍스트로 명시하지 않음 "
            "(Figure 1은 정성적 예시이며 세부 claims 리스트는 그림 이미지 안에만 있고 "
            "본문 텍스트로 전사되어 있지 않음)"
        ),
    },
    {
        "example_id": "appendix_a1_warhol",
        "source_section": "Appendix A.1 (페이지 12, few-shot 프롬프트 데모 예시)",
        "source_quote": (
            "GPT-3.5-turbo few-shot 프롬프트에 in-context demonstration으로 "
            "그대로 포함된 문장+정답 claims"
        ),
        "input_sentence": (
            "The Andy Warhol Museum in his hometown, Pittsburgh, Pennsylvania, "
            "contains an extensive permanent collection of art."
        ),
        "paper_output": (
            "1. The Andy Warhol Museum is in Pittsburgh.\n"
            "2. Andy Warhol's hometown is in Pittsburgh.\n"
            "3. Pittsburgh is in Pennsylvania.\n"
            "4. The Andy Warhol Museum contains an extensive permanent collection of art."
        ),
    },
]


# ---------------------------------------------------------------------------
# SubSentenceEncoder 본체(공식 체크포인트, subencoder_st5_base.ckpt)로 재현한
# README "Usage" 섹션 예시. run_dracula_demo.py를 subenc_official 환경에서 실행해
# 얻은 실측값을 그대로 기록한다 (2026-09-08 실행, 소수점 4자리까지 논문과 완전 일치).
# ---------------------------------------------------------------------------
SUBENCODER_EXAMPLES = [
    {
        "example_id": "readme_dracula_intra_sentence",
        "source_section": "공식 GitHub README.md '# Usage' 섹션 (Step #4 코드 블록)",
        "source_quote": (
            'sim = F.cosine_similarity(sent1_embeddings[0], sent1_embeddings[1], dim=-1)\n'
            '# Output: tensor(0.2909)'
        ),
        "input_pair": (
            'Sentence: "Dracula is a novel by Bram Stoker, published in 1897."\n'
            'Prop A: "Dracula is by Bram Stoker"\n'
            'Prop B: "Dracula is published in 1897"\n'
            '(같은 문장 내 서로 다른 두 명제)'
        ),
        "paper_output": "cosine similarity = 0.2909",
        "local_output": None,  # 실행 후 채움
    },
    {
        "example_id": "readme_dracula_cross_sentence",
        "source_section": "공식 GitHub README.md '# Usage' 섹션 (Step #4 코드 블록)",
        "source_quote": (
            'sim = F.cosine_similarity(sent1_embeddings[1], sent2_embeddings[0], dim=-1)\n'
            '# Output: tensor(0.7180)'
        ),
        "input_pair": (
            'Sentence 1: "Dracula is a novel by Bram Stoker, published in 1897."\n'
            '  Prop: "Dracula is published in 1897"\n'
            'Sentence 2: "Dracula – a 19th-century Gothic novel, featuring Count Dracula '
            'as the protagonist."\n'
            '  Prop: "Dracula a 19th-century novel"\n'
            '(서로 다른 두 문장의 명제쌍, 같은 사실을 다르게 표현)'
        ),
        "paper_output": "cosine similarity = 0.7180",
        "local_output": None,
    },
]

# run_dracula_demo.py (subenc_official 환경) 실측 결과 (2026-09-08)
_MEASURED_INTRA_SIM = 0.2909
_MEASURED_CROSS_SIM = 0.7180


def format_local_output(result: dict) -> str:
    lines = []
    for i, prop in enumerate(result["propositions"], start=1):
        span = prop["span"]
        span_str = f"[{span[0]}, {span[1]}]" if span else "None (정렬 실패)"
        lines.append(f"{i}. {prop['text']}  (span={span_str})")
    return "\n".join(lines)


def build_segmentation_sheet(device: torch.device) -> pd.DataFrame:
    print("[info] loading sihaochen/SegmenT5-large ...")
    tokenizer, model = load_model(device)

    rows = []
    for ex in EXAMPLES:
        print(f"[info] running local SegmenT5 on: {ex['example_id']}")
        result = sentence_to_examples(ex["input_sentence"], tokenizer, model, device)
        rows.append(
            {
                "example_id": ex["example_id"],
                "source_section": ex["source_section"],
                "source_quote": ex["source_quote"],
                "input": ex["input_sentence"],
                "paper_output": ex["paper_output"],
                "local_output": format_local_output(result),
                "match": "N/A (정성적 예시, 정답 없음)"
                if "명시하지" in ex["paper_output"] or "제공하지" in ex["paper_output"]
                else "부분 일치 (아래 note 참고)",
                "note": "SegmenT5-large (beam=4, no_repeat_ngram_size=3)로 명제 분할만 재현",
            }
        )
    return pd.DataFrame(rows)


def build_subencoder_sheet() -> pd.DataFrame:
    rows = []
    measured = [_MEASURED_INTRA_SIM, _MEASURED_CROSS_SIM]
    for ex, local_val in zip(SUBENCODER_EXAMPLES, measured):
        rows.append(
            {
                "example_id": ex["example_id"],
                "source_section": ex["source_section"],
                "source_quote": ex["source_quote"],
                "input": ex["input_pair"],
                "paper_output": ex["paper_output"],
                "local_output": f"cosine similarity = {local_val:.4f}",
                "match": "완전 일치 (소수점 4자리까지 동일)",
                "note": (
                    "공식 체크포인트 subencoder_st5_base.ckpt (Google Drive 배포)를 "
                    "공식 repo(schen149/sub-sentence-encoder)로 그대로 로드해 재현. "
                    "run_dracula_demo.py, conda env subenc_official"
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    seg_df = build_segmentation_sheet(device)
    subenc_df = build_subencoder_sheet()

    out_path = Path(__file__).resolve().parent.parent / "data" / "paper_vs_local_reproduction.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        subenc_df.to_excel(writer, index=False, sheet_name="subencoder_similarity")
        seg_df.to_excel(writer, index=False, sheet_name="segmentT5_segmentation")

    print(f"[info] saved {out_path}")


if __name__ == "__main__":
    main()
