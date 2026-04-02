---
description: Python processing.py ↔ C# FlashHSI 런타임 패리티 검증
agent: build
subtask: true
---

`Python_Analysis/models/processing.py`의 순수 수학 함수들이 C# FlashHSI 런타임과 수학적으로 동일한 결과를 내는지 검증합니다.

## 배경

HSI_ML_Analyzer는 두 파트로 구성됩니다:
- **Part 1 (이 레포)**: Python/PyQt5 오프라인 학습기 → `model.json` 출력
- **Part 2 (C# FlashHSI 레포)**: >250 ~ 700 FPS 실시간 런타임 — `model.json` 소비

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

## 검증 절차

1. `Python_Analysis/models/processing.py` 전체 내용 읽기
2. `Python_Analysis/services/learning_service.py`의 `export_model()` 읽기 (`RequiredRawBands`, `PrepChainOrder` 계산 로직)
3. `Python_Analysis/services/processing_service.py`의 `get_base_data()` + `apply_preprocessing_chain()` 읽기 (Absorbance 위치, 체인 순서 확인)
4. 각 함수별 위 표와 대조하여 수식 확인
5. **LogGapFeatureExtractor 부호/공식** 집중 검토:
   - `processing_mode`가 `"Absorbance"`일 때 Python 파이프라인 순서 추적
   - `ProcessingService.get_base_data()` + `apply_preprocessing_chain()`에서 `apply_absorbance` 와 `apply_simple_derivative` 순서 확인
   - C# `LogGapFeatureExtractor.Extract()`가 `log10(valTarget / valGap)`인지 확인
6. `apply_snv()`가 `ProcessingService.process_cube()`나 워커에서 실제 호출되는지 grep
7. `RequiredRawBands` 계산 로직이 gap + order와 정합하는지 확인
8. `HsiPipeline.RegisterProcessorsByChainOrder()`가 `PrepChainOrder`를 재현하는지 확인
9. 잠재적 패리티 위험 목록 출력
10. `DerivOrder > 1` 모델인 경우 C# 런타임의 동등 차수 미분 재현 여부를 별도 확인 (미확인 시 ⚠️)

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
