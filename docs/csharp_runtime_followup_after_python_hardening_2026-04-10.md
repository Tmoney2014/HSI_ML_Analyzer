# C# Runtime Follow-Up Brief After Python Hardening (2026-04-10)

> 목적: `docs/python_runtime_contract_hardening_brief_2026-04-10.md`에서 Python export contract를 먼저 정리한 뒤, 그 안정화된 계약을 기준으로 FlashHSI C# 런타임에서 무엇을 수정해야 하는지 정리한 후속 handoff 문서.  <!-- AI가 수정함: Python 선행 후 C# 후속 문서 신규 생성 -->
> 전제: Python contract hardening이 먼저 수행되거나, 최소한 export shape / field semantics가 문서로 고정되어 있어야 한다.  <!-- AI가 수정함: 문서 간 의존관계 명시 -->

---

## 1. Why This Document Is Separate

최근 변경은 Python과 C# 양쪽 모두에 영향을 주지만, 실제로는 **Python contract를 먼저 고정해야 C# 구현 범위도 불필요한 재작업 없이 확정**된다.  <!-- AI가 수정함: 문서 분리 이유 설명 -->

따라서 본 문서는 다음을 전제로 한다.  <!-- AI가 수정함 -->

1. Python `export_model()`의 **모델별** `Weights` / `Bias` policy가 결정됨  <!-- AI가 수정함: 모델별 contract 전제 명확화 -->
2. `docs/inference_runtime_spec.md`가 최신 `model.json` 필드에 맞게 업데이트됨  <!-- AI가 수정함 -->
3. `RequiredRawBands`, `PrepChainOrder`, `EstimatedFPS` 관련 regression tests가 Python 쪽에 추가됨  <!-- AI가 수정함 -->

---

## 2. Confirmed FlashHSI Target Files

`.opencode/commands/parity-check.md` 기준으로 직접 확인된 C# 대상 파일은 다음 다섯 개다.  <!-- AI가 수정함: 근거 있는 파일만 유지 -->

1. `C:\Users\user16g\Desktop\FlashHSI\FlashHSI.Core\ModelData.cs`  <!-- AI가 수정함 -->
2. `C:\Users\user16g\Desktop\FlashHSI\FlashHSI.Core\Pipelines\HsiPipeline.cs`  <!-- AI가 수정함 -->
3. `C:\Users\user16g\Desktop\FlashHSI\FlashHSI.Core\Preprocessing\RawGapFeatureExtractor.cs`  <!-- AI가 수정함 -->
4. `C:\Users\user16g\Desktop\FlashHSI\FlashHSI.Core\Preprocessing\LogGapFeatureExtractor.cs`  <!-- AI가 수정함 -->
5. `C:\Users\user16g\Desktop\FlashHSI\FlashHSI.Core\Preprocessing\Processors.cs`  <!-- AI가 수정함 -->

---

## 3. C# Follow-Up Depends on Python Deliverables

아래 Python deliverable이 먼저 정리되어야 C# 작업 범위도 정확히 고정된다.  <!-- AI가 수정함 -->

### Required Python inputs  <!-- AI가 수정함 -->
1. 최신 `model.json` field list  <!-- AI가 수정함 -->
2. `OriginalType` / `IsMultiClass` 기준의 모델별 `Weights` / `Bias` contract  <!-- AI가 수정함 -->
3. `SelectedBands` / `RequiredRawBands` index contract (0-based, canonical order)  <!-- AI가 수정함 -->
4. `PrepChainOrder` semantics  <!-- AI가 수정함 -->
5. `EstimatedFPS` classification (metadata only)  <!-- AI가 수정함 -->
6. updated `docs/inference_runtime_spec.md`  <!-- AI가 수정함 -->

즉, C# planning은 Python hardening 결과를 **input contract**로 받는 형태여야 한다.  <!-- AI가 수정함 -->

---

## 4. File-by-File C# Follow-Up Checklist

### 4.1 `FlashHSI.Core/ModelData.cs`

**Goal**  <!-- AI가 수정함 -->
- stabilized Python `model.json`을 robust하게 로드  <!-- AI가 수정함 -->

**Expected follow-up after Python hardening**  <!-- AI가 수정함 -->
1. Python이 최종 확정한 top-level field를 전부 수용  <!-- AI가 수정함 -->
2. `OriginalType` / `IsMultiClass`를 기준으로 모델별 `Weights` / `Bias` contract를 Python 문서 기준으로 구현  <!-- AI가 수정함 -->
3. `SelectedBands`를 canonical feature order로 사용  <!-- AI가 수정함 -->
4. `RequiredRawBands`를 authoritative raw-band contract로 사용  <!-- AI가 수정함 -->
5. `OriginalType`가 `RidgeClassifier`, `LogisticRegression`이어도 정상 수용  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P0  <!-- AI가 수정함 -->

**Acceptance criteria**  <!-- AI가 수정함 -->
- Python hardening 이후 생성된 최신 sample `model.json`을 예외 없이 load 가능  <!-- AI가 수정함 -->
- 모델별 binary/multiclass parsing behavior가 Python spec과 일치  <!-- AI가 수정함 -->

---

### 4.2 `FlashHSI.Core/Pipelines/HsiPipeline.cs`

**Goal**  <!-- AI가 수정함 -->
- Python export가 정의한 preprocessing order와 runtime execution order를 맞춤  <!-- AI가 수정함 -->

**Expected follow-up after Python hardening**  <!-- AI가 수정함 -->
1. `PrepChainOrder`가 있으면 그 순서대로 processor registration  <!-- AI가 수정함 -->
2. 없으면 documented fallback order 사용  <!-- AI가 수정함 -->
3. `Raw masking -> mode conversion -> ordered preprocessing` 순서 유지  <!-- AI가 수정함 -->
4. `Absorbance` 중복 적용 금지  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P0  <!-- AI가 수정함 -->

**Acceptance criteria**  <!-- AI가 수정함 -->
- Python exporter test fixture와 C# runtime이 동일 processor order를 구성  <!-- AI가 수정함 -->
- mode / prep ordering mismatch가 없음  <!-- AI가 수정함 -->

---

### 4.3 `FlashHSI.Core/Preprocessing/RawGapFeatureExtractor.cs`

**Goal**  <!-- AI가 수정함 -->
- Python `apply_simple_derivative()`와 수학적 패리티 유지  <!-- AI가 수정함 -->

**Required checks**  <!-- AI가 수정함 -->
1. 방향이 `Band[i+gap] - Band[i]`인지 확인  <!-- AI가 수정함 -->
2. gap overflow 시 fail-fast  <!-- AI가 수정함 -->
3. `DerivOrder > 1` 지원 여부가 Python policy와 모순 없는지 확인  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P0  <!-- AI가 수정함 -->

---

### 4.4 `FlashHSI.Core/Preprocessing/LogGapFeatureExtractor.cs`

**Goal**  <!-- AI가 수정함 -->
- Python absorbance+derivative path와 수학적 패리티 유지  <!-- AI가 수정함 -->

**Required checks**  <!-- AI가 수정함 -->
1. 방향이 `log10(target/gap)`인지 확인  <!-- AI가 수정함 -->
2. epsilon handling이 Python documented behavior와 동일한지 확인  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P0  <!-- AI가 수정함 -->

---

### 4.5 `FlashHSI.Core/Preprocessing/Processors.cs`

**Goal**  <!-- AI가 수정함 -->
- SNV / MinMax / L2 / SG behavior를 Python 정책과 동기화  <!-- AI가 수정함 -->

**Required checks**  <!-- AI가 수정함 -->
1. SNV: `ddof=1`, zero-guard `1e-9`  <!-- AI가 수정함 -->
2. MinMax: `range > 1e-9` guard  <!-- AI가 수정함 -->
3. L2: `sumSq > 1e-9` equivalent guard  <!-- AI가 수정함 -->
4. SG parameter validation이 Python policy와 일치하는지 확인  <!-- AI가 수정함 -->

**Priority**  <!-- AI가 수정함 -->
- P1  <!-- AI가 수정함 -->

---

## 5. Recommended Sequence

### Stage 0: Python first  <!-- AI가 수정함 -->
- Python export contract hardening 수행  <!-- AI가 수정함 -->
- spec / README / tests 업데이트  <!-- AI가 수정함 -->

### Stage 1: C# schema & loader sync  <!-- AI가 수정함 -->
- `ModelData.cs`  <!-- AI가 수정함 -->

### Stage 2: C# pipeline order sync  <!-- AI가 수정함 -->
- `HsiPipeline.cs`  <!-- AI가 수정함 -->

### Stage 3: C# preprocessing parity sync  <!-- AI가 수정함 -->
- `RawGapFeatureExtractor.cs`  <!-- AI가 수정함 -->
- `LogGapFeatureExtractor.cs`  <!-- AI가 수정함 -->
- `Processors.cs`  <!-- AI가 수정함 -->

### Stage 4: Cross-runtime parity verification  <!-- AI가 수정함 -->
- Python fixture vs C# runtime numeric comparison  <!-- AI가 수정함 -->

---

## 6. Cross-Runtime Acceptance Test Plan

Python hardening 완료 후, 아래 parity test를 C# follow-up acceptance criteria로 사용한다.  <!-- AI가 수정함 -->

1. 동일 sample `model.json` 로딩 성공  <!-- AI가 수정함 -->
2. 동일 raw sample에 대해 `RequiredRawBands` 해석 결과가 Python과 동일  <!-- AI가 수정함 -->
3. 동일 `PrepChainOrder`에서 processor order가 동일  <!-- AI가 수정함 -->
4. derivative output이 Python과 동일 부호/값을 가짐  <!-- AI가 수정함 -->
5. LogGap output이 Python과 동일 부호/값을 가짐  <!-- AI가 수정함 -->
6. SNV / MinMax / L2 numeric output이 허용 오차 내에서 동일  <!-- AI가 수정함 -->
7. `OriginalType`별 binary / multiclass model score path가 Python export contract와 모순되지 않음  <!-- AI가 수정함 -->

---

## 7. Prometheus Prompt Seed (Python Then C#)

```text
Python-side contract hardening 이후를 전제로, FlashHSI C# 런타임 follow-up implementation plan을 수립해줘.

선행 문서:
- docs/python_runtime_contract_hardening_brief_2026-04-10.md
- docs/inference_runtime_spec.md (Python hardening 후 최신화된 버전)
- .opencode/commands/parity-check.md

대상 파일:
- FlashHSI.Core/ModelData.cs
- FlashHSI.Core/Pipelines/HsiPipeline.cs
- FlashHSI.Core/Preprocessing/RawGapFeatureExtractor.cs
- FlashHSI.Core/Preprocessing/LogGapFeatureExtractor.cs
- FlashHSI.Core/Preprocessing/Processors.cs

핵심 목표:
- Python stabilized model.json contract를 C#에서 정확히 수용
- PrepChainOrder 기반 processor order 재현
- RequiredRawBands를 authoritative contract로 사용
- `OriginalType` + `IsMultiClass` 기준 모델별 parser/score path 구성
- RawGap / LogGap / SNV / MinMax / L2 parity 확보

원하는 출력:
- 파일별 작업 항목
- P0/P1 우선순위
- 구현 순서
- acceptance criteria
- Python fixture 기반 cross-runtime parity test plan
- unknowns / risks
```

---

## 8. Notes

- 이 문서는 Python hardening 이후의 C# follow-up 문서다.  <!-- AI가 수정함 -->
- Python contract가 고정되기 전에는 C# parser/pipeline 구현도 재작업 위험이 있다.  <!-- AI가 수정함 -->
