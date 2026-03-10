---
description: models/processing.py 함수의 C# 런타임 패리티 검증
agent: build
subtask: true
---

`Python_Analysis/models/processing.py`의 순수 수학 함수들이 C# WPF 런타임과 동일한 결과를 내는지 검증합니다.

## 배경

HSI_ML_Analyzer는 두 파트로 구성됩니다:
- **Part 1 (이 레포)**: Python/PyQt5 오프라인 학습기 → `model.json` 출력
- **Part 2 (별도 C# 레포)**: 700 FPS 실시간 런타임 — `model.json` 소비

`models/processing.py`의 함수들은 C# 런타임과 **수학적으로 동일**해야 합니다. 패리티가 깨지면 학습 결과가 런타임에서 다르게 동작합니다.

## 검증 대상 함수

다음 함수들을 집중 검토합니다:

1. `apply_simple_derivative(cube, gap)` — Gap-Diff (Log-Gap-Diff 핵심)
   - 공식: `result[b] = cube[b+gap] - cube[b]` (각 픽셀)
   - 출력 밴드 수: `B - gap`

2. `apply_rolling_3point_depth(cube)` — Continuum Removal Lite
   - 공식: `depth[b] = cube[b] - (cube[b-1] + cube[b+1]) / 2`
   - 경계 처리: 양끝 밴드 제외 (출력 밴드 수: `B - 2`)

3. `apply_absorbance(cube)` — 반사율 → 흡광도
   - 공식: `-log10(R)`, `R ≤ 0`이면 클리핑 필요

4. `apply_snv(cube)` — Standard Normal Variate
   - **주의**: MROI 비호환 — 프로덕션에서 사용 금지 확인

5. SG (Savitzky-Golay) 관련 함수 (있다면)

## 검증 절차

1. `models/processing.py` 전체 내용 읽기
2. 각 함수의 구현 로직을 수식으로 추출
3. 경계 케이스 확인:
   - 배열 인덱스 off-by-one 오류
   - 분모가 0인 경우 처리
   - NaN/Inf 처리
   - 밴드 수 계산 정확성
4. `RequiredRawBands` 계산 로직이 `apply_simple_derivative`의 gap과 일치하는지 확인
   - `services/learning_service.py`의 `export_model()` 참조
5. 잠재적 패리티 위험 목록 출력

## 출력 형식

```
=== C# 패리티 검증 보고서 ===

[함수명]
- 구현 수식: ...
- 출력 밴드 수: ...
- 경계 처리: ...
- 위험 요소: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[RequiredRawBands 계산]
- 현재 로직: ...
- Gap=N일 때 필요한 밴드: ...
- 일치 여부: ✅ / ❌

[종합 의견]
...
```

## 주의사항
- 실제 파일 수정 금지 (읽기 전용 분석)
- C# 코드는 이 레포에 없으므로 Python 구현만 기준으로 분석
- `apply_snv()`가 어딘가에서 호출되고 있다면 경고
