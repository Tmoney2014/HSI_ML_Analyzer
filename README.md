# HyperSort - 고속 HSI 플라스틱 선별 시스템

> 700 FPS 실시간 플라스틱 선별을 위한 **하드웨어 인지형(Hardware-Aware) 오프라인 학습 도구**

## 🎯 프로젝트 개요

이 시스템은 단순한 분류기가 아닙니다. **극한의 속도(700 FPS)**와 **산업 현장의 제약(조명, 대역폭)**을 모두 극복하기 위해 하드웨어와 소프트웨어가 정교하게 맞물린 **고성능 하드웨어 인지형 선별 시스템**입니다.

### 핵심 도전 과제

| 문제 | 해결책 |
|------|--------|
| 154개 밴드 전체 사용 시 700 FPS 불가 | **SPA / 지도형 밴드 선택**으로 소수 핵심 밴드 압축 |
| MROI 환경에서 전체 통계(SNV) 사용 불가 | **Log-Gap-Diff**로 조명 변화 상쇄 |
| 복잡한 비선형 모델(DL, kernel SVM)은 느림 | **선형 모델 + 경량 전처리**로 고속 추론 유지 |

---

## 📋 시스템 구성

```
┌───────────────────────────────────────────────────────────────┐
│ Part 1. 오프라인 학습 (Python) ← 현재 저장소                   │
│   Raw HSI → 밴드 선택 → 전처리 체인 → 선형 모델 학습          │
│                         ↓                                     │
│                   model.json Export                           │
└───────────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────────┐
│ Part 2. 런타임 (C# WPF) ← 별도 저장소                          │
│   model.json 로드 → MROI 카메라 설정 → 700 FPS 추론            │
└───────────────────────────────────────────────────────────────┘
```

---

## 🚀 빠른 시작

### 요구 사항

- Python 3.9+
- PyQt5
- NumPy, scikit-learn, spectral

### 설치

```bash
cd Python_Analysis
pip install -r requirements.txt
```

### 실행

```bash
python main.py
```

---

## 📁 디렉토리 구조

```
Python_Analysis/
├── main.py                      # 진입점
├── config.py                    # 설정 로더
├── settings.json                # 런타임 설정
├── Gemini.md                    # AI 컨텍스트 (상세)
├── models/
│   └── processing.py            # 핵심 전처리 함수
├── services/
│   ├── data_loader.py           # ENVI 파일 로드
│   ├── processing_service.py    # 전처리 파이프라인
│   ├── band_selection_service.py # SPA 알고리즘
│   ├── learning_service.py      # 모델 학습 + Export
│   └── optimization_worker.py   # Auto-ML 최적화
├── viewmodels/                  # MVVM ViewModel
└── views/                       # PyQt5 UI
```

---

## 🔧 주요 기능

### 1. 데이터 관리
- ENVI 형식 HSI 파일 로드
- 그룹별 파일 분류 (PP, PE, ABS 등)
- 스마트 캐싱 시스템

### 2. 분석 & 시각화
- 실시간 스펙트럼 시각화
- 배경 제거 (Threshold + Rules)
- 전처리 파이프라인 (SG, Gap Diff)

### 3. 학습 & 최적화
- SPA, ANOVA-F, SPA-LDA, LDA-coef, Full Band 선택 지원
- LDA, SVM, PLS-DA, Ridge, Logistic Regression 모델 지원
- Auto-ML 최적화 (Gap, NDI, 밴드 수)

### 4. Export
- C# 호환 `model.json` 생성
- MROI용 `RequiredRawBands` 자동 계산

---

## 📄 출력 파일 (model.json)

```json
{
  "ModelType": "LinearModel",
  "OriginalType": "LogisticRegression",
  "IsMultiClass": true,
  "SelectedBands": [40, 42, 97, 99, 105],
  "RequiredRawBands": [40, 42, 45, 47, 97, 99, 102, 104, 105, 110],
  "EstimatedFPS": 732.81,  // (double) Diagnostic estimate only — not a runtime correctness field. See inference_runtime_spec.md.
  "PrepChainOrder": ["SG", "SimpleDeriv", "MinMax"],
  "Weights": [[...], [...]],
  "Bias": [...],
  "Preprocessing": {
    "Mode": "Raw",
    "ApplySG": true,
    "SGWin": 5,
    "SGPoly": 2,
    "SGDeriv": 0,
    "ApplyDeriv": true,
    "Gap": 5,
    "DerivOrder": 1
  },
  "Labels": { "0": "PP", "1": "PE" },
  "Colors": { "0": "#00FF00", "1": "#FF0000" }
}
```

> 참고: `Weights` / `Bias`의 정확한 shape는 `OriginalType`과 `IsMultiClass`에 따라 달라질 수 있습니다.  <!-- AI가 수정함: 모델별 shape contract 안내 -->
>
> `PrepChainOrder`는 C# 런타임이 Python 전처리 순서를 재현하기 위한 우선 계약이고, `EstimatedFPS`는 진단/표시용 metadata입니다.  <!-- AI가 수정함: runtime contract 의미 보강 -->

### model.json 필드 분류 (Field Classification)

| 필드 | 카테고리 | 설명 |
|------|----------|------|
| `ModelType`, `OriginalType`, `IsMultiClass` | Runtime-Critical | 추론 타입 결정 |
| `SelectedBands`, `RequiredRawBands` | Runtime-Critical | MROI 카메라 밴드 설정 |
| `Weights`, `Bias` | Runtime-Critical | 선형 분류기 파라미터 |
| `Preprocessing` | Runtime-Critical | 전처리 파라미터 |
| `PrepChainOrder` | Runtime-Critical | 전처리 순서 계약 (C# 패리티 필수) |
| `Labels`, `Colors` | Runtime-Critical | 클래스 → 레이블/색상 매핑 |
| `EstimatedFPS`, `ModelName`, `Description`, `Timestamp`, `Performance` | Optional metadata | 진단/표시용 |

---

## 🤖 AI 개발자용

AI 코딩 도구(Gemini, Copilot 등)를 사용하는 경우:
- **[Gemini.md](Python_Analysis/Gemini.md)** - 상세 컨텍스트 문서 참조

---

## 📜 라이선스

Private - All Rights Reserved
