---
description: HSI 프로젝트 컨벤션으로 커밋 생성 (type 영어, 나머지 한국어)
agent: build
subtask: true
---

현재 스테이징 상태와 변경사항을 분석하고, HSI_ML_Analyzer 프로젝트 컨벤션에 맞는 커밋을 생성합니다.

## 컨벤션 규칙
- 커밋 타입은 영어: `feat`, `fix`, `refactor`, `docs`, `chore`, `style`, `test`, `perf`
- 나머지 설명은 한국어
- 형식: `<type>: <한국어 설명>`
- 예시: `feat: 학습 결과 캐시 무효화 로직 추가`, `fix: PLS-DA export 속성 누락 버그 수정`

## 작업 지시

$ARGUMENTS

## 절차

1. `git status`로 스테이징 상태 확인
2. `git diff --staged`로 변경 내용 확인 (미스테이징 파일은 `git diff`도 확인)
3. 변경 내용을 분석하여 적절한 커밋 타입 결정
4. 위 컨벤션에 맞는 커밋 메시지 초안 작성
5. 사용자가 `$ARGUMENTS`에 메시지를 제공했다면 그것을 우선 사용
6. `git add -A`로 전체 스테이징 후 커밋 실행 (단, `.env`, `venv/`, `__pycache__/`, `*.pyc` 제외)
7. 커밋 성공 여부 확인

## 주의사항
- `Python_Analysis/venv/`는 절대 커밋하지 않음
- `__pycache__/`, `*.pyc` 파일 제외
- `.env` 또는 인증 정보 파일 포함 시 경고 후 중단
