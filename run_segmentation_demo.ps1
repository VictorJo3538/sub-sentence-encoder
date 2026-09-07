# 공식 SegmenT5-large (sihaochen/SegmenT5-large)로 문장 -> 명제 자동 분할 데모.
# 실행: 터미널에서 .\run_segmentation_demo.ps1 또는 탐색기에서 우클릭 -> "PowerShell로 실행"
#
# 이 conda 환경(subenc)은 SegmenT5 체크포인트가 요구하는 torch>=2.6 전용으로 새로 만들었다.
# (llm_safety 환경은 torch 2.5.1이라 train.py/evaluate.py 스모크 테스트에만 사용)

$ErrorActionPreference = "Stop"

$CondaPython = "C:\ProgramData\Anaconda3\envs\subenc\python.exe"

if (-not (Test-Path $CondaPython)) {
    Write-Host "conda 환경(subenc)을 찾을 수 없습니다: $CondaPython" -ForegroundColor Red
    Write-Host "README.md의 '실제 논문 체크포인트 연동' 섹션을 참고해 환경을 먼저 만드세요." -ForegroundColor Yellow
    exit 1
}

Set-Location $PSScriptRoot

Write-Host "SegmenT5-large로 예시 문장들을 명제로 분할합니다 (최초 실행 시 ~3GB 모델 다운로드)..." -ForegroundColor Cyan

& $CondaPython scripts/segment_propositions.py --sentence "Dracula is a novel by Bram Stoker featuring Count Dracula as the protagonist."
& $CondaPython scripts/segment_propositions.py --sentence "The Andy Warhol Museum in his hometown, Pittsburgh, Pennsylvania, contains an extensive permanent collection of art."

Write-Host "`n완료. 직접 문장을 넣어보려면:" -ForegroundColor Green
Write-Host '  & "C:\ProgramData\Anaconda3\envs\subenc\python.exe" scripts/segment_propositions.py --sentence "여기에 문장"'
