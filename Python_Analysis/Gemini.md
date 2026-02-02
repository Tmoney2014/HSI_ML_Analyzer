# HSI ML Analyzer - AI 컨텍스트 문서

> 이 문서는 AI가 프로젝트를 빠르게 이해하기 위한 컨텍스트 파일입니다.

## 1. 프로젝트 개요

**목적:** 초분광 이미지(HSI) 분석 및 머신러닝 모델 학습 도구
**기술 스택:** Python 3.x, PyQt5, NumPy, scikit-learn
**아키텍처:** MVVM (Model-View-ViewModel)

---

## 2. 디렉토리 구조

```
Python_Analysis/
├── main.py                 # 진입점
├── models/
│   └── processing.py       # 핵심 전처리 함수 (순수 함수)
├── services/
│   ├── data_loader.py      # HSI 파일 로드 (ENVI 포맷)
│   ├── processing_service.py # 전처리 파이프라인 통합
│   ├── learning_service.py # 모델 학습/익스포트
│   ├── band_selection_service.py # SPA 알고리즘
│   ├── training_worker.py  # 학습 백그라운드 워커
│   ├── optimization_worker.py # 최적화 워커
│   └── optimization_service.py # Auto-ML 로직
├── viewmodels/
│   ├── main_vm.py          # 상태 관리, SmartCache
│   ├── analysis_vm.py      # 분석 탭 VM
│   └── training_vm.py      # 학습 VM
└── views/                  # UI 컴포넌트
```

---

## 3. 핵심 데이터 흐름

```
[Raw HSI Cube] (H, W, Bands)
      ↓
[Mode 변환] ─── Raw: 그대로
             ├── Reflectance: (S-D)/(W-D)
             └── Absorbance: -log₁₀(Ref)
      ↓
[Masking] ─── Threshold + Rules (Mean, Median 등)
      ↓
[전처리 Chain] ─── SG, SNV, SimpleDeriv, 3PointDepth 등
      ↓
[SPA Band Selection] ─── 최적 밴드 선택
      ↓
[Model Training] ─── SVM, LDA, PLS-DA
```

---

## 4. 주요 클래스 역할

### ViewModels

| 클래스 | 역할 |
|--------|------|
| `MainViewModel` | 파일 그룹 관리, Reference 캐싱, SmartCache (LRU 20개) |
| `AnalysisViewModel` | 분석 탭 상태, 전처리 체인, 스펙트럼 시각화 |
| `TrainingViewModel` | 학습 오케스트레이션, 캐시 무효화 관리 |

### Services

| 클래스 | 역할 |
|--------|------|
| `ProcessingService` | **전처리 파이프라인 단일 소스** |
| `LearningService` | 모델 학습 (SVM/LDA/PLS-DA), JSON 익스포트 |
| `TrainingWorker` | 백그라운드 학습, 3단계 캐시 |
| `OptimizationService` | Auto-ML (Gap, NDI, Bands 최적화) |

---

## 5. 캐시 시스템 (3단계)

| 레벨 | 캐시 | 내용 | 무효화 조건 |
|------|------|------|------------|
| L1 | `data_cache` | Raw Cube | 파일 목록에서 제거 시 |
| L2 | `cached_optimization_base_data` | Masked + Ref 변환 | 설정 변경 시 |
| L3 | `cached_training_data` | Preprocessed | 설정 변경 시 |

**무효화 트리거:** `params_changed`, `refs_changed`, `files_changed`, `mode_changed`
**L1 정리:** `remove_files_from_group()` 호출 시 해당 파일 캐시 삭제

---

## 6. 전처리 함수 (models/processing.py)

| 함수 | 설명 |
|------|------|
| `create_background_mask()` | 배경 마스킹 |
| `apply_mask()` | 마스크 적용 → (N, Bands) |
| `apply_snv()` | Standard Normal Variate |
| `apply_savgol()` | Savitzky-Golay 필터 |
| `apply_simple_derivative()` | Gap Difference (NDI 포함) |
| `apply_rolling_3point_depth()` | Continuum Removal Lite |
| `apply_absorbance()` | -log₁₀(R) 변환 |

---

## 7. 핵심 원칙

1. **분석 탭 스펙트럼 = 학습 데이터** (반드시 동일해야 함)
2. **ProcessingService가 단일 소스** (중복 로직 금지)
3. **Mode 변환 순서:** Raw → Ref → Abs (순서 중요!)
4. **결정론적 순서:** `sorted(valid_groups)` 사용

---

## 8. 수정 시 주의사항

> [!CAUTION]
> - `models/processing.py` 함수는 순수 함수 유지
> - `ProcessingService.convert_to_ref()` 수정 시 학습에 영향
> - 캐시 무효화 시그널 연결 확인 필수

---

## 9. 자주 사용하는 파일 경로

- 전처리 로직: `services/processing_service.py`
- 학습 로직: `services/training_worker.py`, `services/learning_service.py`
- 상태 관리: `viewmodels/main_vm.py`
- UI 분석 탭: `views/tabs/tab_analysis.py`
