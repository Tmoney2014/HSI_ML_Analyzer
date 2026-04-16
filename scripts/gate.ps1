#Requires -Version 5.1
<#
.SYNOPSIS
    HSI_ML_Analyzer 품질 게이트 스크립트
.DESCRIPTION
    lint (ruff) → test (pytest) → typecheck (mypy) 순서로 실행.
    하나라도 실패하면 즉시 중단하고 exit 1.
    반드시 프로젝트 루트에서, venv가 활성화된 상태로 실행할 것.
.NOTES
    전제 조건: Python_Analysis\venv\Scripts\Activate.ps1 실행 후 사용
    빌드(pyinstaller)는 일상 게이트에서 제외 — 릴리스 시에만 실행
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptRoot

# Working directory: always project root
Push-Location $ProjectRoot
try {

    # ── Step 1: Lint (ruff) ───────────────────────────────────────────
    Write-Host ""
    Write-Host "=== [1/3] LINT (ruff) ===" -ForegroundColor Cyan
    python -m ruff check Python_Analysis/ tests/
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[GATE FAILED] Lint 오류가 있습니다. 위 목록을 수정하세요." -ForegroundColor Red
        exit 1
    }
    Write-Host "[PASS] Lint 통과" -ForegroundColor Green

    # ── Step 2: Tests (pytest) ────────────────────────────────────────
    Write-Host ""
    Write-Host "=== [2/3] TEST (pytest) ===" -ForegroundColor Cyan
    python -m pytest tests/ -x --tb=short
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[GATE FAILED] 테스트 실패. 위 출력을 확인하세요." -ForegroundColor Red
        exit 1
    }
    Write-Host "[PASS] 테스트 통과" -ForegroundColor Green

    # ── Step 3: Type check (mypy) ─────────────────────────────────────
    Write-Host ""
    Write-Host "=== [3/3] TYPECHECK (mypy) ===" -ForegroundColor Cyan
    python -m mypy Python_Analysis/ --no-error-summary
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[GATE FAILED] 타입 오류가 있습니다. 위 목록을 수정하세요." -ForegroundColor Red
        exit 1
    }
    Write-Host "[PASS] 타입 검사 통과" -ForegroundColor Green

    # ── All steps passed ─────────────────────────────────────────────
    Write-Host ""
    Write-Host "==============================" -ForegroundColor Green
    Write-Host " 게이트 통과 — 커밋 가능합니다 " -ForegroundColor Green
    Write-Host "==============================" -ForegroundColor Green
    Write-Host ""

} finally {
    Pop-Location
}

exit 0
