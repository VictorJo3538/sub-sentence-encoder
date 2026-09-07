# Sub-Sentence Encoder — 경량 재현 구현 (8GB VRAM 기준)

원 논문: Chen et al., *Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic
Representations*, NAACL 2024. (arXiv:2311.04335)

이 저장소는 논문 전체(문장 분할용 SegmenT5, PropSegmEnt 전체 데이터셋, 다운스트림 응용 실험 등)를
재현하는 것이 아니라, **논문의 핵심 기여인 "마스킹 기반 다중 명제 pooling + 명제 단위 대조학습"** 만을
검증하기 위한 최소 구현입니다. 8GB급 GPU 한 대에서 학습·추론이 가능하도록 설계했습니다.

## 무엇이 포함되어 있는가

- `sub_sentence_encoder/model.py` — bi-encoder 위에 토큰 마스크 기반 mean pooling과
  projection MLP를 얹은 `SubSentenceEncoder` 모듈 (논문 3.1절, Figure 2에 해당).
- `sub_sentence_encoder/losses.py` — in-batch negative를 사용하는 InfoNCE 대조학습 손실
  (논문 3.2절 학습 목적식에 해당하는 최소 구현).
- `sub_sentence_encoder/dataset.py` — (문장 A, 명제 A 마스크, 문장 B, 명제 B 마스크) 쌍 형태의
  JSONL 데이터를 배치로 토크나이징/마스크 생성.
- `train.py` — 학습 루프 (fp16, gradient accumulation 기본 적용).
- `evaluate.py` — 학습된 인코더로 명제 쌍 검색(retrieval) 정확도를 확인하는 간단한 평가 스크립트.
- `configs/base.yaml` — 8GB VRAM 기준 기본 하이퍼파라미터.
- `data/toy_propositions.jsonl` — 파이프라인 동작 확인용 장난감 데이터 10개.
- `scripts/get_real_data.md` — 실제 PropSegmEnt 데이터로 교체하는 방법 안내.
- `scripts/segment_propositions.py` — 논문 저자가 공개한 **실제 체크포인트** `sihaochen/SegmenT5-large`
  (HF Hub)로 문장을 명제로 자동 분할하는 스크립트 (논문 §3.3 Step 1을 직접 학습 대신 공식 체크포인트로 대체).
- `sub_sentence_encoder/align.py` — 논문 Appendix A.3 방식(NLTK lemma affinity + Hungarian algorithm)으로
  패러프레이징된 명제 텍스트를 원문 character span에 정렬 (§3.3 Step 3의 재구현).
- `run_smoke_test.ps1` — `llm_safety` conda 환경으로 toy 데이터 학습+평가를 한 번에 실행하는 원클릭 스크립트.
- `run_segmentation_demo.ps1` — `subenc` conda 환경으로 SegmenT5 자동 명제 분할을 시연하는 원클릭 스크립트.
- `data/paper_vs_local_reproduction.xlsx` — 논문이 명시한 input/output과 로컬 재현 결과를 비교한 표
  (아래 "논문 재현 검증" 섹션 참고).
- `scripts/build_paper_comparison.py` — 위 xlsx를 생성하는 스크립트.
- (이 디렉토리 밖) `../sub-sentence-encoder-official/` — 논문 저자의 **공식 GitHub 저장소**를 그대로
  clone한 것. SUBENCODER 본체 공식 체크포인트(`checkpoints/subencoder_st5_base.ckpt`, 사용자가 Google
  Drive에서 직접 다운로드)를 로드해 결과를 재현하는 데 사용 (`run_dracula_demo.py`).

## 원 논문과의 차이 (의도적으로 축소한 부분)

| 항목 | 원 논문 | 이 구현 |
|---|---|---|
| 인코더 | 다양한 사전학습 인코더로 실험 | `bert-base-uncased` 고정 (경량화) — 학습(`train.py`)은 직접 구현 |
| 명제 분할 | 별도 학습된 SegmenT5로 자동 분할 | 저자가 공개한 **공식 체크포인트** `sihaochen/SegmenT5-large`를 그대로 연동 (`scripts/segment_propositions.py`) |
| 명제 ⇒ span 정렬 | NLTK lemma affinity + Hungarian algorithm | 동일 방식으로 재구현 (`sub_sentence_encoder/align.py`) |
| SUBENCODER 본체 체크포인트 | 저자가 Google Drive에 `.ckpt`로 배포 (HF Hub 아님, 커스텀 PyTorch Lightning 클래스 필요) | 미연동. 대신 이 저장소의 `SubSentenceEncoder`로 직접 학습 |
| 데이터셋 | PropSegmEnt 전체 | 데이터 형식만 호환, 소량 toy 데이터로 스모크 테스트 |
| 음성 예시 | in-batch + 동일 문장 내 다른 명제까지 포함 | in-batch negative만 기본 적용 (동일 문장 내 negative는 옵션) |

> **업데이트(2026-09-08)**: SUBENCODER 본체 공식 체크포인트(`subencoder_st5_base.ckpt`, Google Drive
> 배포)도 실제로 연동해 검증했습니다. `../sub-sentence-encoder-official/`에 공식 저장소를 clone하고
> `subenc_official`이라는 별도 conda 환경(`pytorch_lightning==1.8.5`)을 만들어 로드한 뒤, 공식
> README의 "Usage" 예시(Dracula 문장 코사인 유사도)를 재현한 결과 **0.2909 / 0.7180으로 논문과
> 소수점 4자리까지 완전히 일치**했습니다. `data/paper_vs_local_reproduction.xlsx`의
> `subencoder_similarity` 시트 참고.

## 논문 재현 검증 (paper_vs_local_reproduction.xlsx)

`data/paper_vs_local_reproduction.xlsx`에 "논문이 명시한 input/output"과 "우리가 로컬에서 실제로
실행한 output"을 나란히 비교한 표를 만들어 두었습니다. 2개 시트로 구성:

- **`subencoder_similarity`** — SubSentenceEncoder 본체(공식 체크포인트)로 재현. 논문 공식 GitHub
  README의 Dracula 예시 코사인 유사도(0.2909, 0.7180)를 완전히 재현 확인.
- **`segmentT5_segmentation`** — SegmenT5-large(공식 체크포인트)로 재현한 문장→명제 분할 3개 예시
  (§1 Dracula, Figure 1, Appendix A.1 Andy Warhol Museum). Appendix A.1만 논문이 정답 4-way 분할을
  텍스트로 명시하고 있어 직접 비교 가능하며, 로컬은 2-way로 뭉치는 등 완전히 같지는 않음 (SegmenT5
  생성 모델 자체의 편차 — 파이프라인 연동 자체는 정상 동작).

재생성하려면 (SegmenT5는 `subenc` 환경, SubSentenceEncoder 실측값은 `subenc_official` 환경에서
`run_dracula_demo.py`로 미리 얻어 스크립트에 반영되어 있음):

```powershell
& "C:\ProgramData\Anaconda3\envs\subenc\python.exe" scripts/build_paper_comparison.py
```

## 바로 실행하기 (원클릭)

이 컴퓨터에는 이미 `llm_safety`라는 conda 환경(`C:\ProgramData\Anaconda3\envs\llm_safety`)에
`torch 2.5.1+cu121`, `transformers 5.15.0`이 설치되어 있어 이를 그대로 사용하도록 맞춰 두었습니다.
(GPU: RTX 3060 Ti 8GB, CUDA 13.2 드라이버 — README의 "8GB VRAM" 가정과 일치)

터미널에서:

```powershell
cd "C:\Users\victo\Documents\동국대\연구실\LLM_Safety\experiment\sub_sentence_encoder"
.\run_smoke_test.ps1
```

또는 탐색기에서 `run_smoke_test.ps1`을 우클릭 → "PowerShell로 실행".
학습(toy 데이터, 3 epoch) → 평가(Recall@1)까지 자동으로 수행하고 `checkpoints/toy_run`에 체크포인트를 저장합니다.
(2026-09-08 기준 이 환경에서 정상 동작 확인함: Recall@1 = 1.0000, toy 데이터라 수치 자체보다 "에러 없이 완주"가 중요합니다.)

### 공식 SegmenT5로 명제 자동 분할 데모

`sihaochen/SegmenT5-large`는 safetensors가 아닌 구형 체크포인트라 `torch>=2.6`이 필요합니다.
`llm_safety`는 torch 2.5.1이라 이 프로젝트 전용으로 `subenc`라는 conda 환경을 새로 만들어
`torch 2.6.0+cu124`를 설치해 두었습니다 (`C:\ProgramData\Anaconda3\envs\subenc`).

```powershell
.\run_segmentation_demo.ps1
```

논문 Figure 1 / Appendix A.1과 동일한 예시 문장 2개를 SegmenT5로 명제 분할하고,
각 명제를 원문 character span에 정렬한 결과를 JSON으로 출력합니다.
직접 문장을 넣으려면:

```powershell
& "C:\ProgramData\Anaconda3\envs\subenc\python.exe" scripts/segment_propositions.py --sentence "여기에 문장"
```

(2026-09-08 기준 정상 동작 확인. 단, SegmenT5 자체의 생성 품질은 문장에 따라 편차가 있어
가끔 명제가 뭉치거나 단어가 어색하게 반복될 수 있습니다 — 이는 파이프라인 문제가 아니라
원 모델의 한계입니다.)

다른 환경으로 새로 설치하려면 아래를 참고하세요.

## 설치 (다른 환경 / 새 conda env 기준)

```powershell
conda create -n subsent python=3.10 -y
conda activate subsent
pip install -r requirements.txt
```

GPU가 CUDA 11.8/12.1 중 어떤 버전인지에 따라 `torch` 설치 커맨드가 달라질 수 있습니다.
`pip install -r requirements.txt` 실행 전, https://pytorch.org/get-started/locally/ 에서
본인 CUDA 버전에 맞는 설치 커맨드를 먼저 실행해 두는 것을 권장합니다.
새 환경을 쓰는 경우 `run_smoke_test.ps1` 상단의 `$CondaPython` 경로를 새 환경 경로로 바꾸거나,
아래처럼 직접 실행하세요.

## 스모크 테스트 (toy 데이터로 파이프라인 확인, 수동 실행)

```powershell
python train.py --config configs/base.yaml --data data/toy_propositions.jsonl --epochs 3 --output_dir checkpoints/toy_run
python evaluate.py --checkpoint checkpoints/toy_run --data data/toy_propositions.jsonl
```

이 단계는 실제 성능 검증이 아니라 "코드가 에러 없이 도는지" 확인하는 용도입니다.

## 8GB VRAM 최적화 팁

- `configs/base.yaml`의 기본값은 `batch_size=16`, `fp16=true`, `grad_accum_steps=2` (실질 배치=32)로
  맞춰 두었습니다. OOM이 나면 `batch_size`를 8로 낮추고 `grad_accum_steps`를 4로 올리세요.
- `bert-base-uncased` 대신 더 가볍게 하려면 `distilbert-base-uncased`로 `configs/base.yaml`의
  `encoder_name`을 바꾸면 됩니다.
- 최대 시퀀스 길이(`max_seq_len`)는 명제가 보통 문장 단위이므로 128이면 충분한 경우가 많습니다.
  더 긴 문단 단위 명제를 다룬다면 256으로 늘리되 batch_size를 줄이세요.

## 실제 데이터로 확장하기

`scripts/get_real_data.md` 참고. PropSegmEnt 데이터셋과 원 논문 공식 코드/체크포인트는
https://github.com/schen149/sub-sentence-encoder/ 에 공개되어 있습니다.

## 인용

```bibtex
@article{chen2023subsentence,
  title={Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations},
  author={Sihao Chen and Hongming Zhang and Tong Chen and Ben Zhou and Wenhao Yu and Dian Yu and Baolin Peng and Hongwei Wang and Dan Roth and Dong Yu},
  journal={arXiv preprint arXiv:2311.04335},
  year={2023}
}
```
