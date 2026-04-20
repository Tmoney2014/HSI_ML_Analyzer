---
description: Python processing.py ↔ C# FlashHSI 런타임 패리티 검증
agent: build
subtask: true
---

`Python_Analysis`가 생성하는 최신 `model.json` 계약과 `models/processing.py` 수학이 C# FlashHSI 런타임에서 동일하게 재현되는지 검증합니다.  <!-- AI가 수정함: Python exporter contract까지 패리티 범위 확장 -->

## 배경

HSI_ML_Analyzer는 두 파트로 구성됩니다:
- **Part 1 (이 레포)**: Python/PyQt5 오프라인 학습기 → `model.json` 출력
- **Part 2 (C# FlashHSI 레포)**: >250 ~ 700 FPS 실시간 런타임 — `model.json` 소비

Python 학습기는 **패리티의 모체(source contract)** 이고, C# FlashHSI는 그 결과(`model.json`)를 소비하는 런타임입니다.  <!-- AI가 수정함: Python-first contract 관계 명시 -->
따라서 패리티 검증은 두 축을 모두 봐야 합니다.  <!-- AI가 수정함 -->

1. `models/processing.py` 및 `ProcessingService`의 수학/순서가 C#에서 동일한가  <!-- AI가 수정함 -->
2. `learning_service.py::export_model()`이 기록한 계약(`OriginalType`, `IsMultiClass`, `RequiredRawBands`, `PrepChainOrder`, `Weights`, `Bias`)을 C#이 올바르게 소비하는가  <!-- AI가 수정함 -->

패리티가 깨지면 학습 결과가 런타임에서 다르게 동작합니다.  <!-- AI가 수정함 -->

## C# 런타임 구현 위치 (FlashHSI)

| Python 함수 | C# 대응 파일 | 클래스/메서드 |
|-------------|-------------|-------------|
| `apply_simple_derivative` | `FlashHSI.Core/Preprocessing/RawGapFeatureExtractor.cs` | `Extract()` |
| `apply_absorbance` + `apply_simple_derivative` | `FlashHSI.Core/Preprocessing/LogGapFeatureExtractor.cs` | `Extract()` |
| `apply_snv` | `FlashHSI.Core/Preprocessing/Processors.cs` | `SnvProcessor.Process()` |
| `apply_minmax_norm` | `FlashHSI.Core/Preprocessing/Processors.cs` | `MinMaxProcessor.Process()` |
| `apply_l2_norm` | `FlashHSI.Core/Preprocessing/Processors.cs` | `L2NormalizeProcessor.Process()` |
| `apply_min_subtraction` | `FlashHSI.Core/Preprocessing/Processors.cs` | `MinSubProcessor.Process()` |
| `apply_savgol` | `FlashHSI.Core/Preprocessing/Processors.cs` | `SavitzkyGolayProcessor.Process()` |
| `apply_absorbance` (standalone) | `FlashHSI.Core/Preprocessing/Processors.cs` | `AbsorbanceProcessor.Process()` |
| (Pipeline 순서) | `FlashHSI.Core/Pipelines/HsiPipeline.cs` | `ProcessFrame()` |
| (model.json 스키마) | `FlashHSI.Core/ModelData.cs` | `ModelConfig`, `PreprocessingConfig` |
| (분류 라우팅) | `FlashHSI.Core/Classifiers/LinearClassifier.cs` | `Predict()` |

## 검증 대상 함수 및 C# 대응 공식

### 1. `apply_simple_derivative(data, gap, order)` ↔ `RawGapFeatureExtractor.Extract()`

**Python 공식** (`processing.py`):
```python
A = data[:, :-gap]   # Index 0 ~ N-gap  (Left / Current)
B = data[:, gap:]    # Index gap ~ N     (Right / Future)
result = B - A       # Band[i+gap] - Band[i]
```

**C# 공식** (`RawGapFeatureExtractor.cs`):
```csharp
// Forward Difference: Band[i+gap] - Band[i]
output[i] = valGap - valTarget;
// where: tIdx = selectedBands[i], gIdx = tIdx + gapShift
```

**검증 포인트:**
- 방향성 일치: Python `B - A` = C# `valGap - valTarget` = `Band[i+gap] - Band[i]` ✔
- 경계 처리: Python은 슬라이싱으로 자동 제거, C#은 Configure 단계에서 gap 인덱스 범위를 검증하고 범위 초과 시 예외 처리(클램프 폴백 없음)
- **검증 포인트**: `RequiredRawBands` + `gapBands` 매핑 불일치는 즉시 실패(fail-fast)해야 함
- **추가 감시 포인트**: Python/C# 모두 현재 `DerivOrder == 1`만 허용됨

### 2. `apply_absorbance` + `apply_simple_derivative` ↔ `LogGapFeatureExtractor.Extract()`

**Python 공식** (`processing.py` + `processing_service.py`):
1. `apply_absorbance`: `R_safe = max(R, epsilon)`, `A = -log10(R_safe)`
2. `apply_simple_derivative`: `Band[i+gap] - Band[i]` (흡광도에 적용)
3. 합산: `(-log10(B)) - (-log10(A))` = `log10(A) - log10(B)` = `log10(A/B)`

**C# 공식** (`LogGapFeatureExtractor.cs`):
```csharp
// Python 패리티 방향: log10(Target/Gap)
output[i] = Math.Log10((valTarget + Epsilon) / (valGap + Epsilon));
// = log10(A/B)
```

**검증 포인트:**
- Python 결과: `log10(A/B)`  (A = target, B = gap)
- C# 결과: `log10(A/B)`
- **현재 기준 수학적 방향 일치** (부호 반전 이슈는 수정 완료 상태여야 정상)
- 만약 C# 코드가 `log10(valGap / valTarget)`로 되어 있으면 즉시 **패리티 위험**으로 분류

### 3. `apply_snv(data)` ↔ `SnvProcessor.Process()`

**Python 공식** (`processing.py`):
```python
std = np.std(data, axis=1, ddof=1, keepdims=True)
valid = std.squeeze(axis=1) > 1e-9
```

**C# 공식** (`Processors.cs`):
```csharp
double std = Math.Sqrt(sumSqDiff / (length - 1));   // ddof=1
if (std > 1e-9) { ... }
```

**검증 포인트:**
- 현재 Python/C# 모두 `ddof=1` 기준
- zero-guard 기준도 `1e-9`로 일치
- `apply_snv()`는 수식 패리티는 확보되어 있으며, 적용 시 데이터셋/도메인(MROI 포함) 검증을 별도로 수행할 것

### 4. `apply_minmax_norm(data)` ↔ `MinMaxProcessor.Process()`

**Python** (`processing.py`): per-spectrum min/max, zero-guard `range > 1e-9`일 때만 정규화
**C#** (`Processors.cs`): per-feature-vector min/max, zero-guard `range > 1e-9`

**✅ 동일**: 둘 다 per-spectrum min-max 정규화. 적용 범위 동일.

### 5. `apply_l2_norm(data)` ↔ `L2NormalizeProcessor.Process()`

**Python** (`processing.py`): `np.linalg.norm`, `sumSq > 1e-9`와 동등한 임계(`norm > sqrt(1e-9)`)에서만 정규화
**C#** (`Processors.cs`): `sqrt(sum of squares)`, `sumSq > 1e-9`일 때만 정규화

**✅ 동일**: L2 Euclidean norm + zero-guard 정책 일치.

### 6. `apply_min_subtraction(data)` ↔ `MinSubProcessor.Process()`  <!-- AI가 수정함: 누락 항목 추가 -->

**Python 공식** (`processing.py`):
```python
return data - np.min(data, axis=1, keepdims=True)
```

**C# 공식** (`Processors.cs`):
```csharp
double minVal = input[0];
for (int i = 1; i < length; i++) if (input[i] < minVal) minVal = input[i];
for (int i = 0; i < length; i++) output[i] = input[i] - minVal;
```

**검증 포인트:**
- 둘 다 per-spectrum (행 단위) 최솟값 차감 ✅
- Python axis=1 keepdims=True = C# 스펙트럼 단위 루프 ✅
- zero-guard 없음 (범위가 0이어도 차감만 수행 — 결과는 전부 0)

### 7. `apply_savgol(data, window_size, poly_order, deriv)` ↔ `SavitzkyGolayProcessor.Process()`  <!-- AI가 수정함: 누락 항목 추가 -->

**Python 공식** (`processing.py`):
```python
# even → odd 보정
if window_size % 2 == 0: window_size += 1
# 검증: window > poly, data.shape[1] >= window, deriv <= poly (ValueError)
result = scipy.signal.savgol_filter(data, window_length=window_size,
                                     polyorder=poly_order, deriv=deriv, axis=1)
```

**C# 공식** (`Processors.cs` / `SavitzkyGolayProcessor`):
```csharp
// even → odd 보정 (동일)
// Vandermonde pseudoinverse로 center coefficients 계산
// ComputeEdgeWeights(): scipy mode='interp' 경계 처리 (polynomial edge interpolation)
// 무효 파라미터: silently clamp (polyOrder → windowSize-1, derivOrder → polyOrder, etc.)
// length < windowSize: 스펙트럼 전체 스킵 (무처리 — 원본 유지)
```

**검증 포인트:**
- 수학: ✅ scipy `savgol_filter` 완전 패리티 (Vandermonde + pseudoinverse)
- 경계 처리: ✅ scipy `mode='interp'` 패리티 (polynomial edge interpolation)
- even→odd 보정: ✅ 동일
- **에러 처리 차이**: Python raises ValueError → C# silently clamps/skips  
  → 허용: Python이 export 전에 유효성 검증 완료. C# 런타임엔 항상 valid params만 도달.
- **주의**: `length < windowSize`일 때 C#은 해당 스펙트럼을 무처리로 통과. 런타임에서 이 조건이 발생하면 SG가 적용되지 않은 스펙트럼이 혼재 가능.

### 8. `apply_absorbance(data, epsilon)` standalone ↔ `AbsorbanceProcessor.Process()`  <!-- AI가 수정함: 누락 항목 추가 -->

**Python 공식** (`processing.py`):
```python
local_R = np.maximum(data, epsilon)   # epsilon=1e-6
return -np.log10(local_R)
```

**C# 공식** (`Processors.cs`):
```csharp
const double Epsilon = 1e-6;
output[i] = -Math.Log10(Math.Max(input[i], Epsilon));
```

**검증 포인트:**
- epsilon 값: Python 1e-6 / C# 1e-6 ✅
- 공식: `-log10(max(R, epsilon))` 완전 일치 ✅
- PrepChain 내 위치: Python에서 `apply_absorbance`는 `get_base_data()` 단계에서 처리되므로 `PrepChainOrder`에 포함되지 않음 (있으면 ValueError). C#은 `AbsorbanceProcessor`가 CalibrationProcessor 다음, PrepChain 이전에 등록됨 — 순서 ✅

### 9. 분류 라우팅 ↔ `LinearClassifier.Predict()`  <!-- AI가 수정함: 누락 항목 추가 -->

**C# 라우팅 분기** (`LinearClassifier.cs`):
| OriginalType 조건 | C# 분기 | 동작 |
|---|---|---|
| `Contains("PLS")` | `PlsThreshold()` | clamp 0~1 → threshold 0.75 → Unknown(-1) 가능 |
| `Contains("SVM")` \| `Contains("SVC")` \| `Contains("Ridge")` | `ArgMaxOnly()` | 항상 클래스 반환 (Unknown 없음) |
| else (LogReg, LDA) | `SoftmaxAndThreshold()` | softmax + 0.75 threshold → Unknown(-1) 가능 |

**검증 포인트:**
- Binary Ridge: `ArgMaxOnly` → 2D weights argmax → Unknown 없음 ✅ (M-1 수정 후)
- Binary LogReg: `SoftmaxAndThreshold` → 2D softmax ≈ sigmoid → 확률 올바름 ✅
- Binary LDA: `SoftmaxAndThreshold` → 2D softmax → ✅
- Binary SVC: `ArgMaxOnly` → ✅
- Binary PLS-DA: `PlsThreshold` (IsMultiClass=False) → 단일 클래스 score clamp → threshold ✅
- **⚠️ 비대칭 주의**: Ridge/SVC binary → Unknown 없음 / LogReg/LDA binary → Unknown(−1) 가능 (0.75 미만 확신도). 의도적 설계이면 OK.

## 추가 계약 검증 포인트 (model-aware contract)  <!-- AI가 수정함: 모델별 shape 계약 섹션 추가 -->

### `learning_service.py::export_model()` ↔ `FlashHSI.Core/ModelData.cs`

**핵심 원칙:**  <!-- AI가 수정함 -->
- Python은 model-specific shape를 유지한다.  <!-- AI가 수정함 -->
- C#은 `OriginalType` + `IsMultiClass`를 먼저 읽고 `Weights` / `Bias` shape를 해석해야 한다.  <!-- AI가 수정함 -->

**필수 확인 필드:**  <!-- AI가 수정함 -->
- `OriginalType`  <!-- AI가 수정함 -->
- `IsMultiClass`  <!-- AI가 수정함 -->
- `SelectedBands`  <!-- AI가 수정함 -->
- `RequiredRawBands`  <!-- AI가 수정함 -->
- `PrepChainOrder`  <!-- AI가 수정함 -->
- `Weights`  <!-- AI가 수정함 -->
- `Bias`  <!-- AI가 수정함 -->

**검증 포인트:**  <!-- AI가 수정함 -->
- `OriginalType`별 `Weights` / `Bias` shape가 Python 문서 계약과 일치하는가  <!-- AI가 수정함 -->
- C# loader가 binary / multiclass를 `IsMultiClass` 기준으로 안정적으로 해석하는가  <!-- AI가 수정함 -->
- `SelectedBands`의 정렬 순서가 C# feature column order와 동일한가  <!-- AI가 수정함 -->
- `RequiredRawBands`가 authoritative raw-band contract로 사용되는가  <!-- AI가 수정함 -->
- `EstimatedFPS`는 metadata로만 취급되고 runtime correctness 판단에 사용되지 않는가  <!-- AI가 수정함 -->

## 검증 절차

1. `Python_Analysis/models/processing.py` 전체 내용 읽기
2. `Python_Analysis/services/learning_service.py`의 `export_model()` 읽기 (`OriginalType`, `IsMultiClass`, `RequiredRawBands`, `PrepChainOrder`, `Weights`, `Bias` 계약 확인)  <!-- AI가 수정함 -->
3. `Python_Analysis/services/processing_service.py`의 `get_base_data()` + `apply_preprocessing_chain()` 읽기 (Absorbance 위치, 체인 순서 확인)
4. 각 함수별 위 표와 대조하여 수식 확인
5. **LogGapFeatureExtractor 부호/공식** 집중 검토:
   - `processing_mode`가 `"Absorbance"`일 때 Python 파이프라인 순서 추적
   - `ProcessingService.get_base_data()` + `apply_preprocessing_chain()`에서 `apply_absorbance` 와 `apply_simple_derivative` 순서 확인
   - C# `LogGapFeatureExtractor.Extract()`가 `log10(valTarget / valGap)`인지 확인
6. `apply_snv()`가 `ProcessingService.process_cube()`나 워커에서 실제 호출되는지 grep
7. `RequiredRawBands` 계산 로직이 gap + order와 정합하는지 확인
8. `HsiPipeline.RegisterProcessorsByChainOrder()`가 `PrepChainOrder`를 재현하는지 확인
9. `ModelData.cs` / 분류기 계층이 `OriginalType` + `IsMultiClass` 기준으로 `Weights` / `Bias`를 해석하는지 확인  <!-- AI가 수정함 -->
10. **SavitzkyGolayProcessor 수식 패리티** 확인:  <!-- AI가 수정함: 누락 항목 추가 -->
    - `ComputeSGCoefficients()` Vandermonde pseudoinverse가 scipy SG coefficients와 수학적으로 동치인지 확인
    - `ComputeEdgeWeights()` 경계 처리가 scipy `mode='interp'` 방식인지 확인
    - even→odd window 보정 로직 동일 여부 확인
    - `length < windowSize` 처리 방식 확인 (스킵 vs 예외)
11. **MinSubProcessor 패리티** 확인:  <!-- AI가 수정함: 누락 항목 추가 -->
    - per-spectrum min 차감 방향 (행 단위) 일치 여부
12. **AbsorbanceProcessor standalone 패리티** 확인:  <!-- AI가 수정함: 누락 항목 추가 -->
    - epsilon 값 (1e-6) 및 `-log10(max(R, eps))` 공식 일치 여부
    - C# pipeline 내 등록 순서 (CalibrationProcessor 다음, PrepChain 이전) 확인
13. **LinearClassifier 분류 라우팅** 확인:  <!-- AI가 수정함: 누락 항목 추가 -->
    - 각 OriginalType별 분기 경로 (ArgMaxOnly / SoftmaxAndThreshold / PlsThreshold) 매핑 확인
    - Binary Ridge가 ArgMaxOnly 경로에 있는지 확인 (M-1 패치 반영 여부)
    - Binary LogReg/LDA의 SoftmaxAndThreshold 0.75 threshold 비대칭 정책 문서화
14. 잠재적 패리티 위험 목록 출력
15. `DerivOrder > 1` 모델인 경우 C# 런타임의 동등 차수 미분 재현 여부를 별도 확인 (미확인 시 ⚠️)

## 출력 형식

```
=== C# 패리티 검증 보고서 (Python ↔ FlashHSI) ===

[apply_simple_derivative ↔ RawGapFeatureExtractor]
- Python 공식: ...
- C# 공식: ...
- 방향 일치: ✅ / ❌
- 경계 처리 차이: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_absorbance + apply_simple_derivative ↔ LogGapFeatureExtractor]
- Python 합산 공식: ...
- C# 공식: log10(valTarget / valGap)
- 수학적 동치 여부: ...
- 부호 반전 위험: ...
- 학습-추론 대칭성 판단: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_snv ↔ SnvProcessor]
- ddof 차이: Python ddof=1 / C# ddof=1
- zero-guard 차이: Python 1e-9 / C# 1e-9
- 프로덕션 실제 사용 여부: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_minmax_norm ↔ MinMaxProcessor]
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_l2_norm ↔ L2NormalizeProcessor]
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_min_subtraction ↔ MinSubProcessor]  <!-- AI가 수정함: 누락 항목 추가 -->
- per-spectrum 방향 일치: ✅ / ❌
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_savgol ↔ SavitzkyGolayProcessor]  <!-- AI가 수정함: 누락 항목 추가 -->
- Vandermonde pseudoinverse 수식 패리티: ✅ / ❌
- scipy mode='interp' 경계 처리: ✅ / ❌
- even→odd 보정: ✅ / ❌
- length < windowSize 처리: C# 스킵 정책 / Python ValueError — 런타임 영향: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_absorbance standalone ↔ AbsorbanceProcessor]  <!-- AI가 수정함: 누락 항목 추가 -->
- epsilon: Python 1e-6 / C# 1e-6
- 공식 일치: -log10(max(R, eps)) ✅ / ❌
- Pipeline 내 위치 (PrepChain 이전): ✅ / ❌
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[분류 라우팅 ↔ LinearClassifier.Predict()]  <!-- AI가 수정함: 누락 항목 추가 -->
- Ridge binary: ArgMaxOnly (Unknown 없음) ✅ / ❌
- LogReg/LDA binary: SoftmaxAndThreshold (0.75 threshold) ✅ / ⚠️
- SVC binary: ArgMaxOnly ✅ / ❌
- PLS-DA binary: PlsThreshold ✅ / ❌
- Ridge→ArgMax 패치 반영 여부: ✅ / ❌
- 비대칭 정책 (Ridge/SVC: Unknown 없음 vs LogReg/LDA: Unknown 가능): 문서화됨 / ⚠️ 미문서
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[RequiredRawBands 계산]
- 현재 로직 (learning_service.py): ...
- Gap=N, Order=M일 때 생성되는 밴드: ...
- C# Clamp와 충돌 가능성: ...
- 일치 여부: ✅ / ❌

[Pipeline 순서 패리티]
- Python (get_base_data + apply_preprocessing_chain 순서): ...
- C# (HsiPipeline.ProcessFrame + RegisterProcessorsByChainOrder): Raw → RawProcessors → Extract → FeatureProcessors
- 순서 일치: ✅ / ⚠️ / ❌

[PrepChainOrder 패리티]
- Python export_model() 출력: PrepChainOrder
- C# LoadModel()/RegisterProcessorsByChainOrder() 재현 여부: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[Model-specific schema parity]  <!-- AI가 수정함: 모델별 shape 계약 진단 섹션 추가 -->
- Python export_model() 출력: OriginalType / IsMultiClass / Weights / Bias
- C# LoadModel() 해석 규칙: ...
- 모델별 shape 일치 여부: ✅ / ⚠️ / ❌

[종합 의견]
- 즉시 수정 필요: ...
- 감시 필요: ...
- 안전 확인됨: ...
```

## 주의사항

- **실제 파일 수정 금지** (읽기 전용 분석)
- C# 코드는 `C:\Users\user16g\Desktop\FlashHSI` 에 위치 — 필요 시 직접 읽기
- `apply_snv()`가 `ProcessingService.process_cube()` 또는 워커에서 실제로 호출되고 있다면 즉시 경고
- LogGap 공식은 현재 기준 `log10(target/gap)`가 정답. 반대 방향 발견 시 즉시 ❌ 위험으로 보고
- `DerivOrder > 1` 모델은 C# 런타임의 차수 재현 경로를 별도 검증하기 전까지 기본적으로 감시 대상으로 분류
- RawGap/LogGap에서 clamp fallback 로그가 발생하면 `RequiredRawBands` 생성/매핑 불일치 가능성으로 즉시 점검
- SavitzkyGolay: `length < windowSize` 런타임 조건 발생 시 C#은 스펙트럼을 무처리로 통과. 배치 안에 혼재 가능성 경고  <!-- AI가 수정함: 누락 주의사항 추가 -->
- Binary 분류 라우팅: Ridge/SVC→ArgMax(Unknown 없음), LogReg/LDA→SoftmaxAndThreshold(0.75 threshold Unknown 가능). 이 비대칭은 의도적 설계이며 M-1 수정(2026-04-20) 이후 상태가 정답임  <!-- AI가 수정함: 누락 주의사항 추가 -->
