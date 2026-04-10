# 논문 연구 계획서
# 흑색 폐플라스틱 초고속 선별을 위한 MWIR 초분광 소프트웨어 정의 다분광 센서 아키텍처

**작성일:** 2026-04-09  
**프로젝트:** HSI_ML_Analyzer (오프라인 학습 도구)  
**연구 상태:** 실험 설계 단계

---

## 1. 논문 개요

### 제목 (안)
> **"Software-Defined Multispectral Sensor Architecture for High-Speed Sorting of Black Waste Plastics Using MWIR Hyperspectral Imaging"**
>
> (흑색 폐플라스틱 초고속 선별을 위한 MWIR 초분광 이미징 기반 소프트웨어 정의 다분광 센서 아키텍처)

### 핵심 Contribution (3가지)

| # | Contribution | 기존 연구와 차별점 |
|---|---|---|
| ① | MWIR + 산업급 700FPS 통합 파이프라인 | MWIR 가능성만 제시(Rozenstein 2017), 산업 속도 미구현 |
| ② | SPA 밴드 → GenICam MROI 역주입 (Software-defined Sensor) | 기존 연구는 소프트웨어 알고리즘만, 하드웨어 제어 미연동 |
| ③ | 이중 노출 + Log-Gap 수학적 스케일 소거 | 이중 노출 보정 + 연산 경량화 동시 달성 논문 없음 |

### 타겟 학술지 (우선순위)
1. **Waste Management** (IF ~8.0) — 흑색 플라스틱 선별 직접 연관
2. **Sensors (MDPI)** (IF ~3.9) — HSI 시스템 논문 다수
3. **Journal of Spectral Imaging** — 초분광 전문
4. **Computers and Electronics in Agriculture** — 고속 선별 응용

---

## 2. 연구 배경 및 문제 정의

### 2.1 왜 흑색 플라스틱이 문제인가
```
기존 SWIR (900~1700nm)
  → 카본블랙이 빛을 흡수
  → 신호 = 0 → 재질 구분 불가

MWIR (2700~5300nm)
  → 고분자 사슬 분자 진동 직접 포착
  → 흑색이어도 재질별 고유 스펙트럼 존재
  → 유일한 물리적 대안
```

### 2.2 MWIR 도입 시 발생하는 새로운 문제
```
문제 1: 데이터 병목
  640픽셀 × 154밴드 × 16bit × 377FPS = 75.4 MB/s
  → 1GigE (125MB/s) 내에 들어오나,
  → 700FPS 목표 달성 불가 (센서 한계 377FPS)
  → MROI로 밴드 수 축소 필수

문제 2: 포화(Saturation) 딜레마
  흑색 플라스틱: 반사율 < 5% → 긴 노출 필요
  백색 레퍼런스: 반사율 높음 → 긴 노출 시 포화
  → 이중 노출 필수 → 스케일 불일치 발생

문제 3: 이중 노출이 만드는 스케일 오차
  k = T_long / T_short (노출 시간 비율)
  Raw_target ∝ k × Raw_white_ref
  → 반사율 계산 왜곡
  → Log-Gap으로 수학적 소거 필요
```

---

## 3. 이론적 배경

### 3.1 SPA (Successive Projections Algorithm)

**목적:** 154개 밴드에서 상호 독립적인 핵심 밴드 N개 추출

**동작 원리:**
```
X = (5000픽셀 × 154밴드) 행렬  ← y(클래스 레이블) 사용 안 함

Step 1: 모든 밴드의 column norm 계산
  norm(밴드k) = √(픽셀1² + 픽셀2² + ... + 픽셀5000²)
  → norm 최대 밴드 선택 (정보량 최대)

Step 2: 선택된 밴드 방향 성분 제거 (직교화)
  P_new = P - v(v^T P / v^T v)
  → 이미 선택된 정보 제거

Step 3: 직교화된 P에서 다시 norm 최대 밴드 선택
Step 4: N개 선택될 때까지 반복

결과: 서로 선형 독립적인 N개 밴드
```

**SPA의 한계:**
```
y를 사용하지 않음 → 클래스 분리 최적화 아님
노이즈 밴드도 norm이 크면 선택될 수 있음
```

### 3.2 ANOVA F-score 기반 밴드 선택

**목적:** 클래스 분리력이 가장 큰 밴드 선택 (y 활용)

```
        클래스 간 분산 (Between-class variance)
F =    ─────────────────────────────────────────
        클래스 내 분산 (Within-class variance)

PP   픽셀: 120, 118, 122 → 뭉침 (내부 분산 작음)
PE   픽셀:  30,  28,  31 → 뭉침 (내부 분산 작음)
ABS  픽셀: 200, 198, 201 → 뭉침 (내부 분산 작음)
PS   픽셀:  80,  79,  81 → 뭉침 (내부 분산 작음)

→ 클래스 간 평균 차이 큼 → F 값 큼 → 선택
```

### 3.3 SPA-LDA 조합 밴드 선택

```
Step 1: SPA로 후보 밴드 M개 추출 (M > N, 넉넉하게)
Step 2: M개 중 LDA 분류 정확도 기준으로 최적 N개 선택
  → Forward selection 또는 exhaustive search
결과: SPA 속도 + LDA 클래스 인식 = 두 장점 결합
```

### 3.4 LDA (Linear Discriminant Analysis)

**런타임 추론 (C# 구현):**
```
입력: 픽셀 스펙트럼 벡터 x (N밴드)
출력: argmax(W·x + b)

연산량: N × C (N=밴드수, C=클래스수)
  N=10, C=4: 40번 곱셈 → 매우 빠름
  N=83, C=4: 332번 곱셈 → 여전히 빠름
```

### 3.5 Log-Gap 전처리 (이중 노출 스케일 소거)

**수학적 증명:**
```
이중 노출 상황:
  White_ref 촬영: 짧은 노출 T_short
  Target 촬영:   긴 노출   T_long
  k = T_long / T_short

Raw 신호:
  raw_target[i]   = k × reflectance[i] × illumination
  raw_target[i+g] = k × reflectance[i+g] × illumination

Log-Gap 연산:
  feature = log(raw_target[i]) - log(raw_target[i+g])
           = log(k × r[i] × I) - log(k × r[i+g] × I)
           = log(k) + log(r[i]) + log(I)
           - log(k) - log(r[i+g]) - log(I)
           = log(r[i]) - log(r[i+g])
           = log(r[i] / r[i+g])

→ 스케일 k 완전 소거
→ 조명 변화 I 완전 소거
→ 순수 반사율 비율만 남음
→ O(1) 연산 (나눗셈 1회 + 로그 2회)
```

---

## 4. 실험 설계

### 4.1 Phase 1: 하드웨어 제약 확정 (카메라 실측)

**목적:** 700FPS 달성 가능한 최대 밴드 수 N 확정

**실험 방법:**
```
Specim FX50 + GenICam으로 MROI 밴드 수 변경하면서 실측 FPS 기록

측정 포인트:
  밴드 수  │  실측 FPS  │  전송량(MB/s)
  ─────────│────────────│─────────────
  154      │  ???       │  계산값
  120      │  ???       │
  100      │  ???       │
   83      │  ???       │  (이론상 700FPS)
   60      │  ???       │
   40      │  ???       │
   20      │  ???       │
   10      │  ???       │

→ FPS vs 밴드수 그래프 작성
→ 700FPS 달성 교차점 = N_max 확정
```

**GenICam 쿼리 코드:**
```python
from harvesters.core import Harvester

h = Harvester()
ia = h.create_image_acquirer(0)
node_map = ia.remote_device.node_map

# 현재 FPS 확인
current_fps = node_map.AcquisitionFrameRate.value

# MROI 밴드 수 설정 후 FPS 재측정
node_map.RegionSelector.value = 'Region0'
node_map.Height.value = N_bands  # 밴드 수 변경
actual_fps = node_map.AcquisitionFrameRate.value

# 센서 클록 (readout speed 계산용)
node_map.DeviceClockSelector.value = 'Sensor'
sensor_clock = node_map.DeviceClockFrequency.value
```

**결과물:** `N_max` — 이후 모든 실험의 상한 제약

---

### 4.2 Phase 2: 밴드 선택 방법 비교 실험

**목적:** N_max개 밴드 제약 안에서 가장 좋은 밴드 선택 방법 찾기

**비교할 밴드 선택 방법:**

| 방법 | y 사용 | 설명 | 구현 상태 |
|------|--------|------|-----------|
| SPA | ❌ | 현재 구현, 직교성 기반 | ✅ 완료 |
| ANOVA-F | ✅ | 클래스 분리력 직접 최적화 | ✅ 완료 |
| SPA-LDA Fast | ✅ | SPA 후보 → LDA coef ranking (빠름) | ✅ 완료 |
| SPA-LDA Greedy | ✅ | Greedy CV 선택 (정확, 느림) | ✅ 완료 |
| LDA-coef | ✅ | LDA 계수 크기 기준 | ✅ 완료 |
| Full Band (baseline) | — | N_max개 전체 사용 | ✅ 완료 |

**실험 프로토콜:**
```
데이터: 흑색 플라스틱 4종 (PP, PE, ABS, PS)
  - 각 클래스 최소 500 샘플 이상
  - Train/Test = 80/20 stratified split
  - 5-fold cross validation

밴드 수 후보: [5, 8, 10, 15, 20, N_max]
  → 각 밴드 수 × 각 선택 방법 조합 실험

평가 지표:
  - Accuracy (전체 정확도)
  - F1-Score (Macro, 클래스 불균형 고려)
  - Precision / Recall per class
  - 선택된 밴드 인덱스 (재현성)
```

---

### 4.3 Phase 3: 분류 모델 비교 실험

**목적:** 선택된 밴드 위에서 최적 분류 모델 찾기

**비교할 모델:**

| 모델 | 특징 | 런타임 적합성 | 현재 구현 |
|------|------|--------------|-----------|
| LDA | Fisher criterion, 행렬곱 1회 | ✅ 최적 | ✅ 완료 |
| Linear SVM | 초평면 최대 마진 | ✅ 적합 | ✅ 완료 |
| PLS-DA | 회귀 기반 판별 | ✅ 적합 | ✅ 완료 |
| Ridge Classifier | L2 정규화 선형 | ✅ 적합 | ✅ 완료 |
| Logistic Regression | 확률 출력 선형 | ✅ 적합 | ✅ 완료 |
| QDA | 비선형 경계 | ⚠️ 느림 | ❌ 비교용만 |

**실험 매트릭스 (최종 결과 테이블):**
```
밴드선택\모델      LDA    SVM    PLS-DA   Ridge   LogReg
───────────────   ─────  ─────  ──────   ─────   ──────
SPA                ???    ???    ???      ???     ???
ANOVA-F            ???    ???    ???      ???     ???
SPA-LDA Fast       ???    ???    ???      ???     ???
SPA-LDA Greedy     ???    ???    ???      ???     ???
LDA-coef           ???    ???    ???      ???     ???
Full Band          ???    ???    ???      ???     ???
```

**구현 메모 (2026-04-10):**
- 현재 HSI_ML_Analyzer에는 `ExperimentRunner` / `ExperimentWorker` 기반의 experiment grid 실행 인프라가 구현되어 있다.  <!-- AI가 수정함: 실제 구현 상태 반영 -->
- Training 탭의 **Export Matrix** UI는 다중 선택 band methods × 다중 선택 model types 조합을 실행하여 aggregate CSV, per-trial CSV, per-trial confusion matrix PNG를 저장한다.  <!-- AI가 수정함: 다중 선택 UI 반영 -->
- 현재 자동화 평가는 train/test split 기반이며, 논문 최종 제출용 5-fold cross validation 및 통계 정리는 후속 실험 정리 단계에서 확정한다.  <!-- AI가 수정함: 현재 구현과 논문 최종 프로토콜 구분 -->

---

### 4.4 Phase 4: 전처리 파이프라인 비교

**목적:** Log-Gap 전처리의 효과 정량적 검증

| 전처리 조합 | 설명 |
|-------------|------|
| Raw only | 전처리 없음 (baseline) |
| Reflectance | 단일 노출 백색 레퍼런스 보정 |
| Log-Gap (단일 노출) | 이중 노출 없이 Log-Gap만 |
| Log-Gap (이중 노출) | **제안 방법** |
| SNV | 표준 정규화 (비교용) |
| Savitzky-Golay | 평활화 (비교용) |

**검증 항목:**
```
① 정확도 변화: 이중 노출 보정 전후
② 스케일 소거 검증:
   - k값 인위 변경 후 정확도 유지 여부
   - k = [0.5, 1.0, 2.0, 5.0] 테스트
③ 연산 시간: 각 전처리 방법별 ms 측정
```

---

### 4.5 Phase 5: 시스템 통합 성능 검증

**목적:** 실제 컨베이어 벨트 환경에서 700FPS + 정확도 동시 달성 검증

**측정 항목:**
```
속도 지표:
  - 실측 FPS (MROI N_max 밴드 설정)
  - 프레임당 처리 시간 (ms)
  - GigaE 실제 사용 대역폭 (MB/s)

정확도 지표:
  - Precision per class (PP/PE/ABS/PS)
  - Recall per class
  - F1-Score (Macro)
  - Confusion Matrix

비교 기준:
  Before MROI: 154밴드, ???FPS
  After MROI:  N_max밴드, 700FPS
  → 속도 향상률 및 정확도 유지 여부
```

---

## 5. 논문 구조 (예상)

```
Abstract
  - 문제: 흑색 플라스틱 SWIR 불가, MWIR 도입 시 속도 병목
  - 방법: SPA+MROI 하드웨어 제어 + 이중노출 Log-Gap 보정
  - 결과: 700FPS 달성, 정확도 ??% (실험 후 기입)

1. Introduction
  1.1 흑색 플라스틱 재활용의 필요성
  1.2 SWIR 한계와 MWIR 대두
  1.3 고속 처리 병목 문제
  1.4 논문 기여 요약

2. Related Work
  2.1 HSI 기반 플라스틱 분류 연구
  2.2 밴드 선택 알고리즘 (SPA, ANOVA, LDA-based)
  2.3 이중 노출 캘리브레이션
  2.4 실시간 HSI 시스템 구현 사례

3. System Architecture
  3.1 전체 파이프라인 개요
       [오프라인 학습] → [model.json] → [런타임 C#]
  3.2 오프라인 학습 모듈 (HSI_ML_Analyzer)
  3.3 런타임 모듈 (C# FlashHSI)
  3.4 GenICam MROI 동적 제어

4. Methodology
  4.1 SPA 기반 최적 밴드 추출
  4.2 밴드 선택 방법 비교 (SPA / ANOVA / SPA-LDA / LDA-coef)
  4.3 이중 노출 캘리브레이션 설계
  4.4 Log-Gap 전처리 수학적 증명
       → k 소거 수식 전개
  4.5 분류 모델 (LDA / SVM / PLS-DA / Ridge)

5. Experiments
  5.1 실험 환경 (Specim FX50, 컨베이어 벨트)
  5.2 데이터셋 (PP/PE/ABS/PS, 샘플 수)
  5.3 Phase 1: FPS vs 밴드수 측정 결과
  5.4 Phase 2: 밴드 선택 방법 비교
  5.5 Phase 3: 분류 모델 비교
  5.6 Phase 4: 전처리 파이프라인 비교
  5.7 Phase 5: 통합 시스템 성능

6. Results & Discussion
  6.1 최적 파이프라인 확정
       (밴드선택방법 + 모델 + 전처리 조합)
  6.2 700FPS 달성 검증
  6.3 이중 노출 보정 효과
  6.4 타 연구와의 비교
  6.5 한계 및 향후 연구

7. Conclusion

References
  - Turner (2018) 흑색 플라스틱
  - Rozenstein (2017) MWIR 흑색 플라스틱
  - Araújo (2001) SPA 알고리즘
  - Rinnan (2009) 전처리 기법 리뷰
  - Barnes (1989) SNV
```

---

## 6. 코드 구현 계획

### 현재 구현된 것 (HSI_ML_Analyzer)

| 기능 | 상태 |
|------|------|
| SPA 밴드 선택 | ✅ 완료 |
| Full Band 모드 | ✅ 완료 |
| ANOVA-F 밴드 선택 | ✅ 완료 |
| SPA-LDA Fast 밴드 선택 | ✅ 완료 |
| SPA-LDA Greedy 밴드 선택 | ✅ 완료 |
| LDA-coef 밴드 선택 | ✅ 완료 |
| LDA 학습/export | ✅ 완료 |
| Linear SVM 학습/export | ✅ 완료 |
| PLS-DA 학습/export | ✅ 완료 |
| Ridge Classifier 학습/export | ✅ 완료 |
| Logistic Regression 학습/export | ✅ 완료 |
| Log-Gap 전처리 (SimpleDeriv) | ✅ 완료 |
| Savitzky-Golay 전처리 | ✅ 완료 |
| model.json → C# 런타임 | ✅ 완료 |

### 추가 구현 필요

| 기능 | 우선순위 | 비고 |
|------|----------|------|
| QDA (비교용) | 🟡 Low | 런타임 미사용, 실험용 |
| 밴드선택 × 모델 자동 그리드 실험 인프라 | ✅ 완료 | `ExperimentRunner`, `ExperimentWorker`, multi-select Export Matrix UI, CSV/Confusion Matrix export 구현 완료 | <!-- AI가 수정함: 실제 구현 상태 반영 -->
| 논문용 5-fold CV 및 최종 통계 테이블 정리 | 🔴 High | 현재 자동화는 train/test split 기반, 논문용 통계 정리 후속 필요 | <!-- AI가 수정함: 남은 연구 작업 재정의 -->
| FPS 실측 로깅 도구 | 🔴 High | Phase 1 필수 |

---

## 7. 데이터 수집 계획

### 필요 샘플

| 클래스 | 최소 샘플 수 | 권장 샘플 수 |
|--------|-------------|-------------|
| PP (흑색) | 500 픽셀 | 2000+ |
| PE (흑색) | 500 픽셀 | 2000+ |
| ABS (흑색) | 500 픽셀 | 2000+ |
| PS (흑색) | 500 픽셀 | 2000+ |

### 촬영 조건 기록 필수
```
- 노출 시간 (T_short, T_long)
- 컨베이어 벨트 속도 (m/s)
- 조명 조건 (on/off, 거리)
- 온도 (MCT 센서 온도 민감)
- 백색 레퍼런스 촬영 간격
```

---

## 8. 예상 결과 및 가설

```
가설 1: ANOVA-F 또는 SPA-LDA가 SPA 단독보다 정확도 높을 것
  근거: y를 활용하므로 클래스 분리에 최적화

가설 2: LDA가 SVM보다 속도 우위, 정확도는 유사할 것
  근거: 행렬곱 1회 vs SVM 서포트 벡터 연산

가설 3: Log-Gap 이중 노출 보정 시 정확도 5~10% 향상
  근거: 스케일 오차 제거로 클래스 간 경계 명확화

가설 4: 최적 밴드 수는 10~20개 (N_max 이하)
  근거: 흑색 플라스틱 고분자별 특징 파장 집중
```

---

## 9. 참고문헌

1. Turner, A. (2018). Black plastics: Linear and circular economies, hazardous additives and marine pollution. *Environment International*, 117, 308-318.
2. Rozenstein, O., Puckrin, E., & Adamowski, J. (2017). Development of a new approach based on midwave infrared spectroscopy for the identification of black plastics. *Waste Management*, 68, 38-44.
3. Araújo, M. C. U. et al. (2001). The successive projections algorithm for variable selection in spectroscopic multicomponent analysis. *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73.
4. Rinnan, Å. et al. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.
5. Barnes, R. J. et al. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777.
6. Fabiyi, S. D. et al. (2021). Folded LDA: Extending the linear discriminant analysis algorithm for feature extraction and data reduction in hyperspectral remote sensing. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*.
7. Kuhn, S. et al. (2023). Systematic reduction of hyperspectral images for high-throughput plastic characterization. *Scientific Reports*.

---

*문서 버전: v0.1 — 실험 결과 취득 후 업데이트 예정*
