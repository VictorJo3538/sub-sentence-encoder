# Sub-Sentence Encoder 원클릭 스모크 테스트
# 실행: 이 파일을 우클릭 -> "PowerShell로 실행" 하거나, 터미널에서 .\run_smoke_test.ps1

$ErrorActionPreference = "Stop"

# 이미 설치되어 있는 conda 환경(llm_safety)의 python을 직접 사용한다.
# (conda가 PATH에 없어도 동작하도록 절대경로 사용)
$CondaPython = "C:\ProgramData\Anaconda3\envs\llm_safety\python.exe"

if (-not (Test-Path $CondaPython)) {
    Write-Host "conda 환경(llm_safety)을 찾을 수 없습니다: $CondaPython" -ForegroundColor Red
    Write-Host "configs/base.yaml 및 이 스크립트 상단의 경로를 본인 환경에 맞게 수정하세요." -ForegroundColor Yellow
    exit 1
}

Set-Location $PSScriptRoot

Write-Host "[1/2] 학습 (toy 데이터, 3 epoch)..." -ForegroundColor Cyan
& $CondaPython train.py --config configs/base.yaml --data data/toy_propositions.jsonl --epochs 3 --output_dir checkpoints/toy_run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[2/2] 평가 (Recall@1)..." -ForegroundColor Cyan
& $CondaPython evaluate.py --checkpoint checkpoints/toy_run --data data/toy_propositions.jsonl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n완료. checkpoints/toy_run 에 체크포인트가 저장되었습니다." -ForegroundColor Green
