---
description: Python processing.py ↔ C# FlashHSI 런타임 패리티 검증
agent: build
subtask: true
---

`Python_Analysis/models/processing.py`의 순수 수학 함수들이 C# FlashHSI 런타임과 수학적으로 동일한 결과를 내는지 검증합니다.

## 배경

HSI_ML_Analyzer는 두 파트로 구성됩니다:
- **Part 1 (이 레포)**: Python/PyQt5 오프라인 학습기 → `model.json` 출력
- **Part 2 (C# FlashHSI 레포)**: >200 FPS 실시간 런타임 — `model.json` 소비

`models/processing.py`의 함수들은 C# 런타임과 **수학적으로 동일**해야 합니다. 패리티가 깨지면 학습 결과가 런타임에서 다르게 동작합니다.

## C# 런타임 구현 위치 (FlashHSI)

| Python 함수 | C# 대응 파일 | 클래스/메서드 |
|-------------|-------------|-------------|
| `apply_simple_derivative` | `FlashHSI.Core/Preprocessing/RawGapFeatureExtractor.cs` | `Extract()` |
| `apply_absorbance` + `apply_simple_derivative` | `FlashHSI.Core/Preprocessing/LogGapFeatureExtractor.cs` | `Extract()` |
| `apply_snv` | `FlashHSI.Core/Preprocessing/Processors.cs` | `SnvProcessor.Process()` |
| `apply_minmax_norm` | `FlashHSI.Core/Preprocessing/Processors.cs` | `MinMaxProcessor.Process()` |
| `apply_l2_norm` | `FlashHSI.Core/Preprocessing/Processors.cs` | `L2NormalizeProcessor.Process()` |
| (Pipeline 순서) | `FlashHSI.Core/Pipelines/HsiPipeline.cs` | `ProcessFrame()` |
| (model.json 스키마) | `FlashHSI.Core/ModelData.cs` | `ModelConfig`, `PreprocessingConfig` |

## 검증 대상 함수 및 C# 대응 공식

### 1. `apply_simple_derivative(data, gap)` ↔ `RawGapFeatureExtractor.Extract()`

**Python 공식** (`processing.py` L151-167):
```python
A = data[:, :-gap]   # Index 0 ~ N-gap  (Left / Current)
B = data[:, gap:]    # Index gap ~ N     (Right / Future)
result = B - A       # Band[i+gap] - Band[i]
```

**C# 공식** (`RawGapFeatureExtractor.cs` L74-75):
```csharp
// Forward Difference: Band[i+gap] - Band[i]
output[i] = valGap - valTarget;
// where: tIdx = selectedBands[i], gIdx = tIdx + gapShift
```

**검증 포인트:**
- 방향성 일치: Python `B - A` = C# `valGap - valTarget` = `Band[i+gap] - Band[i]` ✔
- 경계 처리: Python은 슬라이싱으로 자동 제거, C#은 `gapIdx = Math.Min(tIdx + gapShift, rawBandCount - 1)` Clamp
- **잠재적 불일치**: C#은 Clamp 적용 (범위 초과 시 마지막 밴드 재사용), Python은 슬라이싱이라 해당 경우가 없음. `RequiredRawBands`가 올바르게 설정되어 Clamp가 발생하지 않아야 함.

### 2. `apply_absorbance` + `apply_simple_derivative` ↔ `LogGapFeatureExtractor.Extract()`

**Python 공식** (`processing.py` L241-246 + L151-167):
1. `apply_absorbance`: `R_safe = max(R, epsilon)`, `A = -log10(R_safe)`
2. `apply_simple_derivative`: `Band[i+gap] - Band[i]` (흡광도에 적용)
3. 합산: `(-log10(B)) - (-log10(A))` = `log10(A) - log10(B)` = `log10(A/B)`

**C# 공식** (`LogGapFeatureExtractor.cs` L81-82):
```csharp
// Forward Difference: Log(Gap/Target)
output[i] = Math.Log10((valGap + Epsilon) / (valTarget + Epsilon));
// = log10(B/A)
```

**⚠️ 부호 불일치 위험:**
- Python 결과: `log10(A/B)`  (A = target, B = gap)
- C# 결과: `log10(B/A)`
- **수학적으로 부호가 반대**: `log10(A/B) = -log10(B/A)`
- 그러나 **학습-추론 대칭성** 관점에서: Python으로 학습된 가중치가 이 부호 기준으로 최적화되었다면, C#이 반대 부호를 내더라도 가중치 자체가 반대 방향이라 분류 결과는 동일할 수 있음.
- **실제 패리티 문제 여부**: `apply_absorbance` 모드에서 학습된 모델이 C# 런타임에서 올바른 분류를 하는지 실제 테스트로 확인 필요.

### 3. `apply_snv(data)` ↔ `SnvProcessor.Process()`

**Python 공식** (`processing.py` L77-80):
```python
std = np.std(data, axis=1)   # 기본값: ddof=0 (모표준편차, 분모 N)
```

**C# 공식** (`Processors.cs` L54-55):
```csharp
// Academic Standard: Sample Standard Deviation (N-1)
double std = Math.Sqrt(sumSqDiff / (length - 1));   // ddof=1
```

**⚠️ ddof 불일치:**
- Python: `ddof=0` (모표준편차, 분모 N)
- C#: `ddof=1` (표본표준편차, 분모 N-1)
- **단, `apply_snv()`는 프로덕션 파이프라인에서 사용 금지** (MROI 비호환)이므로 현재 실질 영향 없음.
- 만약 SNV를 사용하는 시나리오가 생기면 Python 측을 `np.std(data, axis=1, ddof=1)` 로 수정 필요.

### 4. `apply_minmax_norm(data)` ↔ `MinMaxProcessor.Process()`

**Python** (`processing.py` L103-107): per-pixel min/max, zero-guard `1e-10`
**C#** (`Processors.cs` L20-37): per-feature-vector min/max, zero-guard `1e-9`

**✅ 동일**: 둘 다 per-spectrum min-max 정규화. 적용 범위 동일.

### 5. `apply_l2_norm(data)` ↔ `L2NormalizeProcessor.Process()`

**Python** (`processing.py` L98-100): `np.linalg.norm`, zero-guard `1e-10`
**C#** (`Processors.cs` L6-17): `sqrt(sum of squares)`, zero-guard `1e-9`

**✅ 동일**: L2 Euclidean norm. Zero-guard 임계값만 미세하게 다름 (실질 차이 없음).

## 검증 절차

1. `Python_Analysis/models/processing.py` 전체 내용 읽기
2. `Python_Analysis/services/learning_service.py`의 `export_model()` 읽기 (RequiredRawBands 계산 로직, L300~L352)
3. `Python_Analysis/services/processing_service.py`의 `process_cube()` 읽기 (파이프라인 순서 확인)
4. 각 함수별 위 표와 대조하여 수식 확인
5. **LogGapFeatureExtractor 부호 이슈** 집중 검토:
   - `processing_mode`가 `"Absorbance"`일 때 Python 파이프라인 순서 추적
   - `ProcessingService.process_cube()` 에서 `apply_absorbance` 와 `apply_simple_derivative` 호출 순서 확인
6. `apply_snv()`가 `ProcessingService.process_cube()`나 워커에서 실제 호출되는지 grep
7. `RequiredRawBands` 계산 로직이 gap + order와 정합하는지 확인
8. 잠재적 패리티 위험 목록 출력

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
- C# 공식: log10(valGap / valTarget)
- 수학적 동치 여부: ...
- 부호 반전 위험: ...
- 학습-추론 대칭성 판단: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_snv ↔ SnvProcessor]
- ddof 차이: Python ddof=0 / C# ddof=1
- 프로덕션 실제 사용 여부: ...
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_minmax_norm ↔ MinMaxProcessor]
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[apply_l2_norm ↔ L2NormalizeProcessor]
- 상태: ✅ 안전 / ⚠️ 주의 / ❌ 위험

[RequiredRawBands 계산]
- 현재 로직 (learning_service.py): ...
- Gap=N, Order=M일 때 생성되는 밴드: ...
- C# Clamp와 충돌 가능성: ...
- 일치 여부: ✅ / ❌

[Pipeline 순서 패리티]
- Python (ProcessingService.process_cube 순서): ...
- C# (HsiPipeline.ProcessFrame 순서): Raw → RawProcessors → Extract → FeatureProcessors
- 순서 일치: ✅ / ⚠️ / ❌

[종합 의견]
- 즉시 수정 필요: ...
- 감시 필요: ...
- 안전 확인됨: ...
```

## 주의사항

- **실제 파일 수정 금지** (읽기 전용 분석)
- C# 코드는 `C:\Users\user16g\Desktop\FlashHSI` 에 위치 — 필요 시 직접 읽기
- `apply_snv()`가 `ProcessingService.process_cube()` 또는 워커에서 실제로 호출되고 있다면 즉시 경고
- LogGap 부호 이슈는 학습-추론 대칭성 관점에서 신중하게 판단 (단순 부호 반전이면 LDA 가중치가 반전 학습되어 실질 패리티 문제 없을 수 있음)
