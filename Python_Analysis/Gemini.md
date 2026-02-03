# HSI ML Analyzer - AI 컨텍스트 문서

> 이 문서는 AI가 프로젝트를 빠르게 이해하기 위한 컨텍스트 파일입니다.

## 1. 프로젝트 개요

**프로젝트명 (가칭):** HyperSort  
**목적:** 700 FPS 실시간 플라스틱 선별을 위한 **하드웨어 인지형(Hardware-Aware) 오프라인 학습 도구**  
**기술 스택:** Python 3.x, PyQt5, NumPy, scikit-learn  
**아키텍처:** MVVM (Model-View-ViewModel)

### 왜 이 프로젝트가 필요한가?

이 시스템은 단순한 분류기가 아닙니다. **극한의 속도(700 FPS)**와 **산업 현장의 제약(조명, 대역폭)**을 모두 극복하기 위해 하드웨어와 소프트웨어가 정교하게 맞물린 시스템입니다.

**핵심 도전 과제:**
- 154개 밴드 전체 사용 시 700 FPS 불가능 → **SPA로 최적 5개 밴드 선택**
- MROI 환경에서 전체 통계(SNV) 사용 불가 → **Log-Gap-Diff로 조명 변화 상쇄**
- 복잡한 모델(SVM, DL)은 느림 → **LDA 단순 행렬 연산으로 99% 정확도**

---

## 2. 시스템 전체 흐름

### Part 1. 오프라인 학습 (Python) ← 현재 프로젝트

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Raw HSI 수집 (Specim FX50)                                   │
│ 2. SPA 밴드 선택 → 최적 5개 직교 밴드                            │
│ 3. Log-Gap-Diff 전처리 → 조명 Scale 상쇄                        │
│ 4. LDA 학습 → 단순 행렬 연산 모델                                │
│ 5. model.json Export → 밴드, Gap, Weights, Threshold             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        model.json
                              ↓
```

### Part 2. 런타임 (C# WPF) ← 별도 프로젝트

```
┌─────────────────────────────────────────────────────────────────┐
│ 런타임 초기화                                                   │
│  - model.json 로드                                              │
│  - MROI 하드웨어 설정 (GenApi)                                  │
│  - 필요한 밴드만 카메라에서 읽도록 설정                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 실시간 선별 루프 (700 FPS)                                      │
│  1.4ms 안에: 배경제거 → Log-Gap-Diff → LDA 추론 → 에어건 발사   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Python 오프라인 학습 파이프라인

```
[Raw HSI Cube] (H, W, 154 Bands)
      ↓
[Mode 변환] ─── Raw: 원본
             ├── Reflectance: (S-D)/(W-D)
             └── Absorbance: -log₁₀(Ref) ← MROI 권장
      ↓
[Masking] ─── Threshold + Rules → 배경 제거
      ↓
[전처리 Chain] ─── SG, SimpleDeriv(Gap Diff), 3PointDepth 등
                   ⚠️ SNV 제외 (MROI에서 전체 통계 못 씀)
      ↓
[SPA Band Selection] ─── 직교 투영으로 중복 없는 최적 밴드 선택
      ↓
[Model Training] ─── LDA (권장), SVM, PLS-DA
      ↓
[model.json Export] ─── 선택 밴드, Gap, Weights, Bias, Threshold
```

---

## 4. 디렉토리 구조

```
Python_Analysis/
├── main.py                     # 진입점
├── config.py                   # 설정 로더
├── settings.json               # 런타임 설정
├── models/
│   └── processing.py           # 핵심 전처리 함수 (순수 함수)
├── services/
│   ├── data_loader.py          # ENVI 파일 로드
│   ├── processing_service.py   # 전처리 파이프라인 통합
│   ├── band_selection_service.py # SPA 알고리즘
│   ├── learning_service.py     # 모델 학습 + JSON Export
│   ├── training_worker.py      # 학습 백그라운드 워커
│   ├── optimization_worker.py  # 최적화 워커
│   └── optimization_service.py # Auto-ML (Gap, NDI, Bands)
├── viewmodels/
│   ├── main_vm.py              # 파일 그룹, SmartCache
│   ├── analysis_vm.py          # 분석 탭 상태
│   └── training_vm.py          # 학습 VM
└── views/                      # PyQt5 UI
```

---

## 5. 핵심 모듈 역할

### 전처리 (MROI 호환 설계)

| 모듈 | 역할 | MROI 호환 |
|------|------|-----------|
| `processing.apply_absorbance()` | -log₁₀(R) 변환 | ✅ |
| `processing.apply_simple_derivative()` | Gap Difference (조명 Scale 상쇄) | ✅ |
| `processing.apply_snv()` | Standard Normal Variate | ❌ 전체 통계 필요 |
| `processing.apply_rolling_3point_depth()` | Continuum Removal Lite | ✅ |

### 밴드 선택 (SPA)

| 함수 | 역할 |
|------|------|
| `band_selection_service.select_best_bands()` | 직교 투영으로 중복 없는 밴드 선택 |

**반환값:**
- `selected_bands`: 선택된 밴드 인덱스 리스트
- `importance_scores`: 초기 Norm (Candidate 시각화용)
- `mean_spectrum`: 평균 스펙트럼

### 학습 & Export

| 클래스 | 역할 |
|--------|------|
| `LearningService.train_model()` | SVM, LDA, PLS-DA 학습 |
| `LearningService.export_model()` | C# 호환 JSON 생성 |

**Export JSON 구조:**
```json
{
  "ModelType": "LinearModel",
  "OriginalType": "LinearDiscriminantAnalysis",
  "SelectedBands": [40, 42, 97, 99, 105],
  "RequiredRawBands": [35, 40, 42, 92, 97, 99, 100, 105],
  "Weights": [[...], [...]],
  "Bias": [...],
  "Preprocessing": { "ApplySG": false, "ApplyDeriv": true, "Gap": 5 },
  "Labels": { "0": "PP", "1": "PE" }
}
```

### Auto-ML 최적화

| 함수 | 역할 |
|------|------|
| `OptimizationService._optimize_gap()` | Gap Size 최적화 |
| `OptimizationService._optimize_ndi()` | NDI Threshold 최적화 |
| `OptimizationService._optimize_bands()` | 밴드 개수 최적화 |

---

## 6. 캐시 시스템

| 레벨 | 캐시 | 내용 | 무효화 조건 |
|------|------|------|-------------|
| L1 | `data_cache` | Raw Cube | 파일 제거 시 |
| L2 | `base_data_cache` | Masked + Ref 변환 | Threshold, Mask Rules 변경 시 |
| L3 | `training_data` | Preprocessed | 전처리 파라미터 변경 시 |

> [!NOTE]
> Gap, SG 등 전처리 파라미터 변경 시 L2는 유지되고 L3만 무효화됩니다.

---

## 7. 핵심 원칙

1. **분석 탭 스펙트럼 = 학습 데이터** (반드시 동일)
2. **ProcessingService가 단일 소스** (중복 로직 금지)
3. **Mode 변환 순서:** Raw → Ref → Abs
4. **MROI 호환 전처리만 사용** (SNV 제외)

---

## 8. 수정 시 주의사항

> [!CAUTION]
> - `models/processing.py` 함수는 순수 함수 유지
> - `ProcessingService.convert_to_ref()` 수정 시 학습에 영향
> - C# 런타임과 JSON 구조 호환성 유지 필수

---

## 9. 설정 파일 (settings.json)

```json
{
  "cache": { "max_items": 20, "min_memory_gb": 1.0 },
  "training": { "default_test_ratio": 0.2, "default_n_features": 5 },
  "model": { "svm_max_iter": 1000 },
  "spa": { "max_samples": 10000 },
  "optimization": { "ndi_step": 100, "ndi_lookahead": 3, "ndi_max_val": 2000 }
}
```

**사용법:**
```python
from config import get, reload_config

value = get('cache', 'max_items', default=20)
reload_config()  # 런타임 리로드
```
