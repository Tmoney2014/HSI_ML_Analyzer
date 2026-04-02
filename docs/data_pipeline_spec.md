# HSI Data Analysis & Learning Pipeline Specification

> **Version**: 1.0  
> **Last Updated**: 2026-02-04  
> **Status**: **Invariant (불변 기준)**

---

## 1. Overview (핵심 요약)

본 문서는 **[파일 로드] → [전처리] → [모델 학습/배포]** 로 이어지는 핵심 데이터 파이프라인의 **불변 규칙(Critical Rules)** 과 **처리 로직(Process Logic)** 을 정의한다.  
이 로직은 **Python Training** 환경과 **C# Runtime** 환경에서 **"반드시 동일하게(Bit-exact)"** 동작해야 한다.

---

## 2. Invariants (절대 원칙)

다음 규칙들은 시스템의 정확성과 일관성을 위해 어떤 경우에도 준수되어야 한다.

### 🛡️ Rule 1. Masking on Raw-Level (마스킹은 원본에서)
배경 제거(Masking)를 위한 임계값 판정(`Thresholding`)은 **반드시 RAW Data (DN 값)** 를 기준으로 수행한다.
- **Why?**: 반사율(`Reflectance`)이나 흡광도(`Absorbance`) 변환 후에는 조명 조건, 노이즈, 로그 연산 등으로 인해 값이 왜곡될 수 있다. 가장 순수한 센서 데이터인 Raw DN이 배경 분리의 가장 확실한 기준이다.
- **Logic**: `Raw > Threshold` (True/False Mask 생성)

### 🛡️ Rule 2. Lazy Processing (전처리는 나중에)
`load_hsi_data` 단계에서는 오직 **파일 읽기**만 수행하며, 어떠한 값의 변형(스케일링, 정규화 등)도 가하지 않는다.
- 데이터 캐싱(`SmartCache`)은 **Base Data (Ref/Abs 변환 + 마스킹 완료, 전처리 미적용)** 상태를 저장한다.
- 전처리(SG, SNV 등)는 항상 사용자가 요청한 시점에 **Base Data에 실시간으로 적용**한다.

### 🛡️ Rule 3. Runtime Compatibility (런타임 호환성)
학습된 모델(`.json`)은 C# 런타임이 **별도의 Python 의존성 없이** 독립적으로 실행할 수 있어야 한다.
- 이를 위해 모델 파일에는 `RequiredRawBands`, `Preprocessing Config` 등 **데이터를 재구성하기 위한 모든 메타데이터**가 포함되어야 한다.
- 학습 시 `Exclude Bands`는 밴드 선택 후보에서 제외되어야 하며, 그 결과가 `SelectedBands`/`RequiredRawBands`에 반영되어야 한다.

---

## 3. Data Pipeline Step-by-Step

### Phase 1: Data Acquisition & Base Preparation
(이 과정은 `LearningService` 및 `TrainingWorker`에서 수행됨)

1.  **File Validation**:
    - 파일 목록이 없는 그룹(`Class`)은 **자동 제외**된다.
    - 예약된 이름(`trash`, `ignore` 등)을 가진 그룹은 제외된다.

2.  **Raw Data Loading**:
    - `.hdr/.bil` 파일을 읽어 Numpy Array `(Height, Width, Bands)` 형태로 로드.
    - `nan_to_num` 처리 필수.

3.  **Base Data Generation** (`ProcessingService.get_base_data`):
    - **Input**: Raw Cube, White Ref, Dark Ref, Threshold
    - **Step A (Mask Check)**: Raw Cube를 기준으로 마스크 생성. (`DN > Threshold`)
    - **Step B (Flattening)**: 마스크가 `True`인 픽셀만 추출 (1D Array).
    - **Step C (Convert)**:
        - `Raw`: 변환 없음.
        - `Reflectance`: `(Raw - Dark) / (White - Dark)`. (0.0 ~ 1.0 Clip)
        - `Absorbance`: `-log10(Reflectance)`. (`Ref <= 0`인 경우 `1e-6`으로 치환 후 계산)
    - **Output**: `Base Data` (전처리가 적용되지 않은, 순수 반사율/흡광도 데이터)

### 마스킹 설정 우선순위 메모 (Runtime 연동)

- Python 학습기는 학습 시점의 `mask_rules/threshold`를 model.json에 기록한다.
- C# 런타임에서는 운영 중 동적 마스킹 설정이 model 값보다 우선될 수 있다.
- 따라서 model.json의 마스킹 정보는 “학습 당시 제안값/폴백값”으로 해석하는 것이 안전하다.

### Phase 2: Preprocessing Pipeline (The Engine)
(사용자 설정에 따라 순차적으로 적용되는 필터 체인. `ProcessingService.apply_preprocessing_chain`)

**권장 순서 (Recommended Order)**:
1.  **Noise Reduction**: `Savitzky-Golay (Smoothing)`
2.  **Baseline Correction**: `Standard Normal Variate (SNV)` or `Min-Max`
3.  **Feature Enhancement**: `Derivatives (1st/2nd)`

> **주의**: `Derivatives` (미분) 적용 시 데이터의 **양 끝단(Edge)** 밴드가 일부 손실(`Gap`)되거나 값이 왜곡될 수 있으므로, **Feature Selection(Band Selection)** 단계에서 이를 고려해야 한다.

### Phase 3: Model Training & Export

1.  **Band Selection (SPA)**:
    - 지정된 `N_features` 개수만큼 최적의 파장(Band)을 선택.
    - 선택된 밴드 인덱스는 **전처리 후의 데이터 기준** 인덱스이다.

2.  **Model Training**:
    - 선택된 밴드(`SelectedBands`)의 데이터로 `LinearSVC` 또는 `PLS-DA` 학습.

3.  **Metadata Packing (Crucial)**:
    - 모델 파일이 C#에서 돌아가기 위해 다음 정보가 필수적으로 포함됨:
        - `InputBands`: 원본 Raw 데이터의 밴드 인덱스 목록 (`RequiredRawBands`).
        - `Preprocessing`: 적용된 전처리 파이프라인 설정값 (`SG Win`, `Poly`, `Deriv Order` 등).
        - `Weights & Bias`: 선형 모델의 계수.
        - `Labels & Colors`: 클래스 이름 및 색상 매핑.

---

## 4. Maintenance Guide (유지보수 가이드)

- **전처리 추가 시**: `ProcessingService.apply_preprocessing_chain`에 로직을 추가하고, `TabAnalysis`의 `default_steps`에 기본 파라미터를 정의한다. **기본값은 `None`이 아닌 Dict 형태여야 한다.**
- **모델 타입 추가 시**: `LearningService.export_model`에서 해당 모델의 가중치(`coef_`)와 편향(`intercept_`)을 추출하는 로직을 추가해야 한다.
- **참조 데이터 변경 시**: `TrainingWorker`는 참조 데이터가 변경되면 캐시를 무효화하고 처음부터 다시 로드해야 한다.
