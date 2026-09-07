"""
공식 논문에서 학습한 SegmenT5-large (sihaochen/SegmenT5-large, HuggingFace Hub 공개)를 이용해
문장을 명제(proposition) 단위로 자동 분할하고, 우리 프로젝트가 쓰는 학습 데이터 형식
(text_a/span_a/text_b/span_b, dataset.py 참고)으로 변환하는 스크립트.

논문 §3.3 "Step 1: Segment Sentences ⇒ Propositions"에 해당하는 부분을,
직접 T5를 학습하는 대신 저자가 공개한 체크포인트를 그대로 사용해서 대체한다.
(SUBENCODER 본체 체크포인트는 Google Drive 수동 배포라 여기서는 다루지 않고,
 SegmenT5만 HF Hub에서 바로 받아 쓴다.)

사용 예:
    # 1) 문장 하나를 명제로 분할해서 확인만 하고 싶을 때
    python scripts/segment_propositions.py --sentence "Dracula is a novel by Bram Stoker, published in 1897."

    # 2) 문장 목록(텍스트 파일, 한 줄에 한 문장)을 분할해서 명제 목록 JSONL로 저장
    python scripts/segment_propositions.py --input_file data/raw_sentences.txt --output_file data/segmented.jsonl

주의:
    - 이 스크립트는 "문장 -> 명제 목록"만 만든다. 학습에 쓰려면 서로 다른 두 문장의 명제끼리
      의미가 같은 쌍(positive pair)을 찾아 text_a/span_a/text_b/span_b 형식으로 만드는 후처리가
      추가로 필요하다 (논문 §3.3 Step 2, NLI 기반 라벨링에 해당 — 이 스크립트 범위 밖).
    - T5-large 모델이라 최초 실행 시 다운로드에 시간이 걸릴 수 있다 (약 3GB).
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sub_sentence_encoder import align_proposition_to_span

MODEL_NAME = "sihaochen/SegmenT5-large"
SEP_TOKEN = "[sep]"


def load_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def segment_sentence(sentence: str, tokenizer, model, device: torch.device) -> list[str]:
    prompt = f"segment sentence: {sentence}"
    input_ids = tokenizer(
        prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True
    ).input_ids.to(device)

    # HF 모델 카드 예제는 num_beams=1(greedy)을 쓰지만, greedy는 이 T5 체크포인트에서
    # 짧은 명제를 반복 생성하는 현상이 관찰되어 beam search + repetition 억제로 대체.
    generated = model.generate(
        input_ids,
        max_new_tokens=256,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    propositions = [p.strip() for p in decoded.split(SEP_TOKEN) if p.strip()]
    return propositions


def find_span(sentence: str, proposition: str) -> list[int] | None:
    """명제 텍스트를 원문 문장에서 character span [start, end)으로 정렬.
    먼저 정확한 부분 문자열 일치를 시도하고(가장 안전), 실패하면 논문 방식
    (NLTK lemma affinity + Hungarian algorithm, sub_sentence_encoder.align 참고)으로
    패러프레이징된 명제도 정렬을 시도한다."""
    idx = sentence.find(proposition)
    if idx != -1:
        return [idx, idx + len(proposition)]
    return align_proposition_to_span(sentence, proposition)


def sentence_to_examples(sentence: str, tokenizer, model, device: torch.device) -> dict:
    props = segment_sentence(sentence, tokenizer, model, device)
    spans = []
    for p in props:
        span = find_span(sentence, p)
        spans.append({"text": p, "span": span})
    return {"sentence": sentence, "propositions": spans}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default=None, help="단일 문장을 바로 분할해서 확인")
    parser.add_argument("--input_file", type=str, default=None, help="한 줄에 한 문장씩 있는 텍스트 파일")
    parser.add_argument("--output_file", type=str, default=None, help="결과를 저장할 JSONL 경로")
    args = parser.parse_args()

    if not args.sentence and not args.input_file:
        parser.error("--sentence 또는 --input_file 중 하나는 필요합니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")
    print(f"[info] loading {MODEL_NAME} ...")
    tokenizer, model = load_model(device)

    if args.sentence:
        result = sentence_to_examples(args.sentence, tokenizer, model, device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    results = []
    for sent in sentences:
        results.append(sentence_to_examples(sent, tokenizer, model, device))

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[info] saved {len(results)} sentences to {args.output_file}")
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
