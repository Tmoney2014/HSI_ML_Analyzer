# Python Runtime Contract Hardening Brief (2026-04-10)

> 목적: 최근 Python 학습기 변경 이후, C# 런타임과의 계약을 더 안정적으로 유지하기 위해 Python 쪽에서 먼저 정리/보강해야 할 항목을 모아둔 handoff 문서.  <!-- AI가 수정함: Python 후속 개선 전용 신규 문서 -->
> 범위: Python exporter / 문서 / 테스트 / contract hardening. C# 수정 자체는 이 문서의 직접 범위가 아니다.  <!-- AI가 수정함: 범위 분리 -->

---

## 1. Executive Summary

현재 Python 쪽은 기능 자체가 크게 잘못되었다기보다, **export contract를 더 단단하게 고정해야 하는 상태**다.  <!-- AI가 수정함: 핵심 요약 -->

가장 중요한 Python-side follow-up은 다음 네 가지다.  <!-- AI가 수정함 -->

1. `export_model()`의 `Weights` / `Bias` shape contract를 **모델별로 명확히 문서화/고정**할 것  <!-- AI가 수정함: shape 통일이 아니라 모델별 계약 명시로 방향 수정 -->
2. `docs/inference_runtime_spec.md`와 `README.md`를 최신 export 현실에 맞게 동기화할 것  <!-- AI가 수정함 -->
3. `RequiredRawBands`, `PrepChainOrder`, `EstimatedFPS`에 대한 regression tests를 추가할 것  <!-- AI가 수정함 -->
4. runtime-facing metadata와 training-only metadata의 경계를 문서상 명확히 할 것  <!-- AI가 수정함 -->

즉, Python의 핵심 과제는 **로직 대수술이 아니라 contract hardening**이다.  <!-- AI가 수정함 -->

---

## 2. Evidence Base

본 문서는 아래 파일들을 근거로 작성되었다.  <!-- AI가 수정함: 근거 명시 -->

- `Python_Analysis/services/learning_service.py`  <!-- AI가 수정함 -->
- `Python_Analysis/services/training_worker.py`  <!-- AI가 수정함 -->
- `Python_Analysis/services/processing_service.py`  <!-- AI가 수정함 -->
- `Python_Analysis/models/processing.py`  <!-- AI가 수정함 -->
- `docs/inference_runtime_spec.md`  <!-- AI가 수정함 -->
- `README.md`  <!-- AI가 수정함 -->
- `tests/` (currently empty)  <!-- AI가 수정함 -->

---

## 3. Python-Side Improvement Areas

### 3.1 `export_model()` shape contract ambiguity

**Observed in**  <!-- AI가 수정함 -->
- `Python_Analysis/services/learning_service.py`  <!-- AI가 수정함 -->

**Current behavior**  <!-- AI가 수정함 -->
- multiclass linear model export:  <!-- AI가 수정함 -->
  - `Weights = [[...], [...], ...]`  <!-- AI가 수정함 -->
  - `Bias = [...]`  <!-- AI가 수정함 -->
- binary linear model export:  <!-- AI가 수정함 -->
  - `Weights = [...]`  <!-- AI가 수정함 -->
  - `Bias = float`  <!-- AI가 수정함 -->
- PLS-DA도 binary/multiclass에 따라 flat/scalar 또는 nested/list 형태가 섞일 수 있다.  <!-- AI가 수정함 -->

**Why this is a Python-side problem**  <!-- AI가 수정함 -->
- 현재 `docs/inference_runtime_spec.md`는 사실상 `Weights=[Class][Feature]`, `Bias=double[]` 쪽으로 읽히며, binary special case와 모델별 차이가 명시되어 있지 않다.  <!-- AI가 수정함 -->
- 이번 프로젝트에서는 shape를 억지로 통일하기보다, **모델별 실제 export shape를 명시적으로 고정하고 C#이 `OriginalType` / `IsMultiClass`를 보고 소비하는 전략**이 더 적합하다.  <!-- AI가 수정함: Option 2를 공식 방향으로 반영 -->
- 따라서 Python 쪽에서 먼저 “각 모델이 어떤 shape를 내보내는지”를 문서와 테스트로 고정해야 한다.  <!-- AI가 수정함 -->

**Chosen Python direction**  <!-- AI가 수정함: 방향 확정 섹션으로 교체 -->
Option 2를 채택한다: **shape를 억지로 통일하지 않고, 모델별 export shape를 유지하되 문서/테스트로 contract를 고정한다.**  <!-- AI가 수정함 -->

> **[2026-04-20 업데이트]**: [C-1] 버그 수정으로 binary LogReg/Ridge/SVC/LDA는
> 2D/1D 형식으로 전환 (Path A 채택). PLS-DA만 예외 (IsMultiClass=false 유지).
> 상세: `.sisyphus/plans/parity-fix.md`

이 방향에서 Python이 해야 할 일:  <!-- AI가 수정함 -->
1. `OriginalType`별 `Weights` / `Bias` shape를 표로 문서화  <!-- AI가 수정함 -->
2. binary / multiclass에서 `IsMultiClass`와 shape 관계를 명시  <!-- AI가 수정함 -->
3. `PLSRegression`(PLS-DA export)의 orientation/예외를 명시  <!-- AI가 수정함 -->
4. C#이 `OriginalType` + `IsMultiClass` 기준으로 안전하게 분기할 수 있도록 sample JSON / tests 제공  <!-- AI가 수정함 -->

이 방향의 장점:  <!-- AI가 수정함 -->
- estimator별 실제 의미를 보존  <!-- AI가 수정함 -->
- Python export 변환 로직을 과도하게 인위화하지 않음  <!-- AI가 수정함 -->
- `OriginalType` metadata를 실질적으로 활용 가능  <!-- AI가 수정함 -->

이 방향의 비용:  <!-- AI가 수정함 -->
- 문서와 테스트가 훨씬 더 정확해야 함  <!-- AI가 수정함 -->
- C# loader가 모델별 분기를 가져야 함  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P0  <!-- AI가 수정함 -->

**Acceptance criteria**  <!-- AI가 수정함 -->
- `OriginalType`별 `Weights` / `Bias` shape contract가 문서와 테스트로 고정됨  <!-- AI가 수정함 -->
- 최신 `model.json` 예시가 모델별 shape policy를 정확히 반영함  <!-- AI가 수정함 -->

---

### 3.2 `docs/inference_runtime_spec.md` drift

**Observed in**  <!-- AI가 수정함 -->
- `docs/inference_runtime_spec.md`  <!-- AI가 수정함 -->

**Drift points confirmed from repo**  <!-- AI가 수정함 -->
- 문서 제목/본문은 `model_config.json` 표기를 사용하고 있음  <!-- AI가 수정함 -->
- `PrepChainOrder` 설명 없음  <!-- AI가 수정함 -->
- `EstimatedFPS` 설명 없음  <!-- AI가 수정함 -->
- `OriginalType`, `ModelName`, `Description`, `Timestamp`, `IsMultiClass` 설명 없음  <!-- AI가 수정함 -->
- binary/multiclass `Weights` / `Bias` variation 설명 없음  <!-- AI가 수정함 -->
- `RidgeClassifier`, `LogisticRegression` linear export 지원 설명 없음  <!-- AI가 수정함 -->
- `RequiredRawBands`가 derivative gap/order + SG radius를 반영한다는 설명 부족  <!-- AI가 수정함 -->

**Why this is Python-side work**  <!-- AI가 수정함 -->
- export contract의 authoritative producer는 Python이므로, contract drift는 먼저 Python 문서에서 정리되어야 한다.  <!-- AI가 수정함 -->

**Required updates**  <!-- AI가 수정함 -->
1. `model.json` 표기로 통일  <!-- AI가 수정함 -->
2. 최신 top-level field 목록 반영  <!-- AI가 수정함 -->
3. `PrepChainOrder` 의미와 fallback policy 명시  <!-- AI가 수정함 -->
4. `EstimatedFPS`를 metadata로 문서화  <!-- AI가 수정함 -->
5. `RequiredRawBands` 계산 의미 강화 설명 추가  <!-- AI가 수정함 -->
6. `OriginalType` / `IsMultiClass` 기준의 모델별 `Weights` / `Bias` contract 명시  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P0  <!-- AI가 수정함 -->

**Acceptance criteria**  <!-- AI가 수정함 -->
- 최신 Python export 결과를 기준으로 spec 문서가 모순 없이 읽힘  <!-- AI가 수정함 -->
- C# 팀이 문서만 읽고도 최신 `model.json`을 구현할 수 있음  <!-- AI가 수정함 -->

---

### 3.3 `README.md` example drift

**Observed in**  <!-- AI가 수정함 -->
- `README.md`의 model.json 예시  <!-- AI가 수정함 -->

**Current limitations**  <!-- AI가 수정함 -->
- `PrepChainOrder` 없음  <!-- AI가 수정함 -->
- `EstimatedFPS` 없음  <!-- AI가 수정함 -->
- `OriginalType` 없음  <!-- AI가 수정함 -->
- `IsMultiClass` 없음  <!-- AI가 수정함 -->
- 최신 linear model 확장 내용 없음  <!-- AI가 수정함 -->

**Why this matters**  <!-- AI가 수정함 -->
- README example은 외부 협업자와 C# 팀이 가장 먼저 보는 quick reference 역할을 한다.  <!-- AI가 수정함 -->

**Required updates**  <!-- AI가 수정함 -->
- 최신 `model.json` 예시 1개 추가  <!-- AI가 수정함 -->
- 최소한 아래 필드 반영  <!-- AI가 수정함 -->
  - `OriginalType`  <!-- AI가 수정함 -->
  - `RequiredRawBands`  <!-- AI가 수정함 -->
  - `EstimatedFPS`  <!-- AI가 수정함 -->
  - `PrepChainOrder`  <!-- AI가 수정함 -->
  - `IsMultiClass`  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P1  <!-- AI가 수정함 -->

---

### 3.4 Missing export regression tests

**Observed in**  <!-- AI가 수정함 -->
- `tests/` directory exists but is empty  <!-- AI가 수정함 -->

**Why this is a gap**  <!-- AI가 수정함 -->
- runtime-facing contract는 exporter에서 바뀌었는데, 자동화된 회귀 테스트가 거의 없다.  <!-- AI가 수정함 -->
- 현재는 채팅/임시 스크립트 기반 검증이 많고, repo-level contract test는 없다.  <!-- AI가 수정함 -->

**Recommended new tests**  <!-- AI가 수정함 -->

#### A. `test_export_model_selected_bands_order.py`  <!-- AI가 수정함 -->
- 비정렬 `selected_bands` 입력 시 export 결과 `SelectedBands`가 정렬되고, `Weights` 열 순서도 그에 맞게 재정렬되는지 검증  <!-- AI가 수정함 -->

#### B. `test_export_model_required_raw_bands.py`  <!-- AI가 수정함 -->
- `SimpleDeriv(gap, order)` + `SG(win)` 조합에 대해 `RequiredRawBands`가 기대값대로 계산되는지 검증  <!-- AI가 수정함 -->
- `total_bands` clamp가 기대대로 작동하는지 검증  <!-- AI가 수정함 -->

#### C. `test_export_model_prep_chain_order.py`  <!-- AI가 수정함 -->
- `PrepChainOrder`가 실제 `preprocessing_config` 순서 그대로 export되는지 검증  <!-- AI가 수정함 -->

#### D. `test_export_model_model_specific_shapes.py`  <!-- AI가 수정함: 테스트명/의도 수정 -->
- `OriginalType`별 binary / multiclass export에서 `Weights`, `Bias`, `IsMultiClass`가 문서화된 계약에 맞게 나오는지 검증  <!-- AI가 수정함 -->

#### E. `test_export_model_estimated_fps.py`  <!-- AI가 수정함 -->
- `EstimatedFPS`가 `RequiredRawBands` 길이를 기준으로 계산되는지 검증  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P0  <!-- AI가 수정함 -->

**Acceptance criteria**  <!-- AI가 수정함 -->
- 위 runtime-facing exporter tests가 CI/로컬에서 재실행 가능  <!-- AI가 수정함 -->
- shape/order/band mapping regressions를 자동으로 잡아낼 수 있음  <!-- AI가 수정함 -->

---

### 3.5 Clarify runtime-facing vs training-only metadata

**Observed in**  <!-- AI가 수정함 -->
- `learning_service.py` export_data에 runtime 필드와 분석/운영용 metadata가 같이 존재  <!-- AI가 수정함 -->

**Examples**  <!-- AI가 수정함 -->
- runtime-critical: `SelectedBands`, `RequiredRawBands`, `Weights`, `Bias`, `Preprocessing`, `PrepChainOrder`  <!-- AI가 수정함 -->
- metadata / optional: `EstimatedFPS`, `Performance`, `ModelName`, `Description`, `Timestamp`, `Colors`, `ExcludeBands`  <!-- AI가 수정함 -->

**Why this matters**  <!-- AI가 수정함 -->
- 문서상 이 경계가 모호하면 C# 팀이 optional metadata를 required contract로 오해할 수 있다.  <!-- AI가 수정함 -->

**Recommended Python-side improvement**  <!-- AI가 수정함 -->
- spec 문서에 field classification 표 추가  <!-- AI가 수정함 -->
  - Required for inference  <!-- AI가 수정함 -->
  - Optional metadata  <!-- AI가 수정함 -->
  - Diagnostic only  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P1  <!-- AI가 수정함 -->

---

### 3.6 Optional improvement: move FPS constants to explicit config policy

**Observed in**  <!-- AI가 수정함 -->
- `learning_service.py`에 `# TODO: move to config in v2` 존재  <!-- AI가 수정함 -->

**Current state**  <!-- AI가 수정함 -->
- FPS estimate는 하드코딩된 카메라 가정값에 의존  <!-- AI가 수정함 -->

**Why this is lower priority**  <!-- AI가 수정함 -->
- 현재 `EstimatedFPS`는 metadata일 뿐, runtime correctness를 직접 깨지 않음  <!-- AI가 수정함 -->

**Recommended action**  <!-- AI가 수정함 -->
- 지금 당장 config로 옮기기보다, 문서에 “diagnostic estimate only”라고 먼저 명시  <!-- AI가 수정함 -->
- 추후 하드웨어 프로파일링 정책이 정해지면 설정화  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P2  <!-- AI가 수정함 -->

---

## 4. Suggested Python Planning Priorities

### P0 (Do first)  <!-- AI가 수정함 -->
- 모델별 export shape contract 결정 및 문서화 (`OriginalType`, `Weights`, `Bias`, `IsMultiClass`)  <!-- AI가 수정함 -->
- `docs/inference_runtime_spec.md` 최신화  <!-- AI가 수정함 -->
- exporter regression tests 추가  <!-- AI가 수정함 -->

### P1 (Do next)  <!-- AI가 수정함 -->
- `README.md` model.json 예시 최신화  <!-- AI가 수정함 -->
- runtime-facing vs optional metadata 분류 표 추가  <!-- AI가 수정함 -->

### P2 (Optional / later)  <!-- AI가 수정함 -->
- `EstimatedFPS` constants 설정화 정책 검토  <!-- AI가 수정함 -->

---

## 5. Deliverables Expected from Python Hardening Work

1. 모델별 exporter contract policy 결정 문서화  <!-- AI가 수정함 -->
2. `docs/inference_runtime_spec.md` 업데이트  <!-- AI가 수정함 -->
3. `README.md` 예시 갱신  <!-- AI가 수정함 -->
4. exporter regression tests 추가  <!-- AI가 수정함 -->
5. C# 팀에 넘길 stabilized contract 요약  <!-- AI가 수정함 -->

---

## 6. Prometheus Prompt Seed (Python First)

```text
최근 HSI_ML_Analyzer Python 학습기 변경 이후, C# 런타임과의 계약을 더 안정적으로 유지하기 위한 Python-side hardening plan을 수립해줘.

목표:
- export_model() contract를 안정화하고
- inference_runtime_spec.md / README.md를 최신 export 현실에 맞게 동기화하고
- runtime-facing exporter regression tests를 추가하는 계획 수립

근거 파일:
- Python_Analysis/services/learning_service.py
- Python_Analysis/services/training_worker.py
- Python_Analysis/services/processing_service.py
- Python_Analysis/models/processing.py
- docs/inference_runtime_spec.md
- README.md
- tests/

핵심 이슈:
- Weights/Bias contract가 모델별로 어떻게 달라지는지 문서상 불명확함
- inference_runtime_spec.md가 PrepChainOrder / EstimatedFPS / OriginalType / IsMultiClass / 최신 model.json 필드를 반영하지 못함
- README model.json example도 최신 export와 어긋남
- tests/가 비어 있어 exporter contract regression test가 부족함

원하는 출력:
- 파일별 작업 항목
- P0/P1/P2 우선순위
- 모델별 export shape contract 표
- 필요한 테스트 목록
- 문서 업데이트 항목
- Python hardening 완료 후 C#에 넘길 stabilized contract 요약
```

---

## 7. Notes

- 이 문서는 Python 쪽 선행 정리를 위한 handoff 문서다.  <!-- AI가 수정함 -->
- C# follow-up은 Python contract hardening 결과를 반영해 별도 계획으로 이어지는 것이 바람직하다.  <!-- AI가 수정함 -->
