# HyperSort - 고속 HSI 플라스틱 선별 시스템

> 700 FPS 실시간 플라스틱 선별을 위한 **하드웨어 인지형(Hardware-Aware) 오프라인 학습 도구**

## 🎯 프로젝트 개요

이 시스템은 단순한 분류기가 아닙니다. **극한의 속도(700 FPS)**와 **산업 현장의 제약(조명, 대역폭)**을 모두 극복하기 위해 하드웨어와 소프트웨어가 정교하게 맞물린 **고성능 하드웨어 인지형 선별 시스템**입니다.

### 핵심 도전 과제

| 문제 | 해결책 |
|------|--------|
| 154개 밴드 전체 사용 시 700 FPS 불가 | **SPA**로 최적 5개 직교 밴드 선택 |
| MROI 환경에서 전체 통계(SNV) 사용 불가 | **Log-Gap-Diff**로 조명 변화 상쇄 |
| 복잡한 모델(SVM, DL)은 느림 | **LDA** 단순 행렬 연산으로 99% 정확도 |

---

## 📋 시스템 구성

```
┌───────────────────────────────────────────────────────────────┐
│ Part 1. 오프라인 학습 (Python) ← 현재 저장소                   │
│   Raw HSI → SPA 밴드 선택 → Log-Gap-Diff 전처리 → LDA 학습    │
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
- 전처리 파이프라인 (SG, Gap Diff, 3-Point Depth)

### 3. 학습 & 최적화
- SPA 기반 밴드 선택
- LDA, SVM, PLS-DA 모델 지원
- Auto-ML 최적화 (Gap, NDI, 밴드 수)

### 4. Export
- C# 호환 `model.json` 생성
- MROI용 `RequiredRawBands` 자동 계산

---

## 📄 출력 파일 (model.json)

```json
{
  "ModelType": "LinearModel",
  "SelectedBands": [40, 42, 97, 99, 105],
  "RequiredRawBands": [35, 40, 42, 92, 97, 99, 100, 105],
  "Weights": [[...], [...]],
  "Bias": [...],
  "Preprocessing": {
    "ApplyDeriv": true,
    "Gap": 5
  },
  "Labels": { "0": "PP", "1": "PE" }
}
```

---

## 🤖 AI 개발자용

AI 코딩 도구(Gemini, Copilot 등)를 사용하는 경우:
- **[Gemini.md](Python_Analysis/Gemini.md)** - 상세 컨텍스트 문서 참조

---

## 📜 라이선스

Private - All Rights Reserved
