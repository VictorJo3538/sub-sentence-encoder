# 실제 데이터로 확장하기

이 프로젝트의 `data/toy_propositions.jsonl`은 파이프라인 동작 확인용 10개짜리 예시입니다.
실제 논문 수준의 검증을 하려면 아래 리소스를 참고해 데이터를 교체하세요.

## 1. 원 논문 공식 코드 / 체크포인트
- https://github.com/schen149/sub-sentence-encoder/
- 이 저장소에 PropSegmEnt 데이터 로딩 방식과 원본 학습 스크립트가 있습니다.
  본 프로젝트의 `SubSentenceEncoder`, `InfoNCELoss` 클래스는 이 공식 구현의 핵심 아이디어를
  최소한으로 재구성한 것이므로, 필요하면 공식 저장소의 데이터 로더를 참고해
  `dataset.py`의 `collate_fn`이 기대하는 형식(`text_a/span_a/text_b/span_b`)으로 변환하면 됩니다.

## 2. PropSegmEnt 데이터셋
- 논문에서 사용한 proposition-level segmentation + entailment 데이터셋입니다.
- 공식 논문(Chen et al., 2023, ACL Findings)과 위 GitHub 저장소에서 다운로드 경로를 확인하세요.
- 원본은 문장을 명제로 분할한 뒤, 서로 다른 문서 간 명제 수준 entailment를 라벨링한 형태입니다.
  본 프로젝트의 형식으로 바꾸려면, entailment(양방향 동치로 볼 수 있는 것)로 표시된 명제 쌍만
  추출해서 `text_a/span_a/text_b/span_b` 형태로 변환하면 됩니다.

## 3. 명제 분할이 필요한 경우 (SegmenT5)
사람이 미리 나눠둔 명제가 없고, 원문 문장에서 자동으로 명제를 분할해야 한다면
아래 모델을 사용할 수 있습니다.

- HuggingFace: `sihaochen/SegmenT5-large`
- 입력 형식: `"segment sentence: {문장}"`
- 출력: 명제들이 `[sep]` 토큰으로 구분되어 나옵니다.

이 모델로 얻은 명제 텍스트를 원문에서 다시 character span으로 매칭시키면
(`str.find()` 등으로) 본 프로젝트의 데이터 형식에 맞출 수 있습니다.
단, `SegmenT5-large`는 T5-large 기반이라 8GB에서 추론은 가능하지만 데이터 전처리 단계에서
별도로 GPU 메모리를 점유하니, 학습과 동시에 돌리지 말고 전처리 단계를 먼저 끝내는 것을
권장합니다.

## 4. 규모를 늘릴 때 8GB에서 주의할 점
- 데이터 양이 늘어도 모델 크기(`bert-base-uncased`)와 `max_seq_len`이 그대로면 VRAM 사용량은
  거의 늘지 않습니다. 늘어나는 것은 학습 시간뿐입니다.
- 배치 안 negative 개수가 성능에 영향을 주는 대조학습 특성상, batch_size를 무리하게 늘리기보다는
  `grad_accum_steps`로 실질적 스텝 수를 늘리는 편이 8GB 환경에서는 더 안전합니다.
