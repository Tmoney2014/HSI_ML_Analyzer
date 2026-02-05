# 🧠 C# Inference Runtime Specification for HSI

> **문서 버전**: v1.0  
> **대상 독자**: C# 런타임 개발자  
> **목적**: Python에서 학습된 모델(`model_config.json`)을 C#에서 로드하고, 실시간(Real-time)으로 추론하기 위한 구현 명세 정의

---

## 1. 개요 (Overview)

이 런타임은 High-Speed Sorter에 탑재되며, Python의 `scikit-learn` 로직을 C#에서 **수동으로 재구현**해야 합니다. (Python 의존성 없음)
핵심은 **Raw Data + Aluminum Background (Inverse Masking)** 전략을 기반으로 한 초고속 추론입니다.

---

## 2. 모델 파일 (`model_config.json`) 구조

### 2.1. JSON Schema

```json
{
  "ModelType": "LinearModel",     // (string) 고정값. Linear SVM 또는 LDA
  "SelectedBands": [10, 25, ...], // (int[]) 학습에 사용된 Feature 밴드 인덱스 (0-based)
  "RequiredRawBands": [5, 10...], // (int[]) 전처리를 위해 로드해야 할 실제 원본 밴드 목록 (C# 최적화용)
  "Performance": { ... },         // (object) 학습 당시 정확도 (참고용)
  
  // 핵심: 선형 모델 계수 (y = Wx + b)
  // Weights: [Class][Feature] 형태의 2차원 배열
  "Weights": [
    [-0.12, 0.45, ... ], // Class 0의 가중치
    [0.05, -0.91, ... ]  // Class 1의 가중치
  ],
  "Bias": [-1.2, 0.5, ...], // (double[]) 각 클래스별 Bias (Intercept)

  // 핵심: 전처리 파이프라인
  "Preprocessing": {
    "Mode": "Raw",          // "Raw", "Reflectance", "Absorbance"
    "ApplyDeriv": true,     // 미분 적용 여부
    "Gap": 5,               // 미분 간격 (중요)
    "DerivOrder": 1,        // 미분 차수 (보통 1)
    "MaskRules": "b80 < 50000", // 마스킹 규칙 (C# 파싱 필요)
    "Threshold": "0.0"      // (Legacy) MaskRule 우선
  },

  "Labels": { "0": "PET", "1": "PP", ... }, // (dict) 클래스 ID -> 이름 매핑
  "Colors": { "0": "#FF0000", ... }         // (dict) 시각화용 색상 코드
}
```

---

## 3. 추론 파이프라인 (Inference Pipeline)

입력으로 들어온 **HSI Line Data** (Width x Bands)를 다음 순서로 처리하여 **Classification Map** (Width)을 생성합니다.

### Step 1: Background Masking (Dynamic Rule)

모델 설정의 `MaskRules` 문자열을 파싱하여 동적으로 마스킹 로직을 적용해야 합니다.
사용자는 검은 배경(`> Threshold`) 또는 알루미늄 배경(`< Threshold`)을 선택할 수 있으므로, 런타임은 부등호를 파싱하여 두 경우를 모두 지원해야 합니다.

*   **Rule Format**: `b{BandIndex} {Operator} {Threshold}` (예: `b80 > 3000` 또는 `b80 < 50000`)
*   **Parsing Logic**:
    1. 접두어 `b` 제거 후 Band Index 파싱.
    2. 연산자 (`>`, `<`, `>=`, `<=`) 파싱.
    3. Threshold 값 파싱.

*   **Logic**:
    ```csharp
    // Example: "b80 < 50000" or "b80 > 3000"
    // Parsed: bandIdx=80, op="<", th=50000.0
    
    double val = pixel[bandIdx];
    bool isObject = false;
    
    if (op == ">") isObject = val > threshold;
    else if (op == "<") isObject = val < threshold;
    
    if (isObject) {
        // Is Object (Valid) -> Proceed to Step 2
    } else {
        // Is Background (Invalid) -> Assign Class -1 (None)
    }
    ```

### Step 2: Preprocessing (Dynamic Chain)

모델 설정 파일(`model_config.json`)의 `Preprocessing` 섹션에 정의된 플래그들을 순서대로 확인하여 적용해야 합니다.
**단, 연산 순서(Order of Operations)는 모델 파일의 JSON 키 순서가 아니라 아래에 정의된 논리적 순서를 따라야 안전합니다.**

#### 1. Data Mode Conversion
카메라로부터 **Raw Data (DN)**가 입력된다고 가정합니다.

*   **Mode: "Raw"**
    *   변환 없음. Raw 값을 그대로 다음 단계로 넘깁니다.

*   **Mode: "Reflectance"**
    *   White/Dark Reference를 사용하여 반사율로 변환합니다.
    *   $\text{Reflectance} = \frac{\text{Raw} - \text{Dark}}{\text{White} - \text{Dark}}$
    *   결과는 0.0 ~ 1.0 사이로 Clip 되어야 합니다.

*   **Mode: "Absorbance"**
    *   **Step A**: 먼저 Reflectance를 계산합니다. (위의 식 참조)
    *   **Step B**: 계산된 Reflectance($R$)에 로그를 취합니다.
    *   $\text{Abs} = -\log_{10}(\max(R, 10^{-6}))$

#### 2. Filtering & Normalization
다음 플래그들이 `true`인 경우 해당 연산을 수행합니다.

*   **Min Subtraction (Baseline Correction)** (`ApplyMinSub`)
    *   각 픽셀의 최솟값을 0으로 맞춥니다.
    *   $x' = x - \min(x)$ (Pixel-wise)

*   **Standard Normal Variate (SNV)** (`ApplySNV`)
    *   각 픽셀 스펙트럼의 평균($\mu$)과 표준편차($\sigma$)를 구합니다.
    *   $x' = \frac{x - \mu}{\sigma}$

*   **Savitzky-Golay Filter (SG)** (`ApplySG`)
    *   파라미터: `SGWin` (Window Size), `SGPoly` (Polynomial Order)
    *   C# 라이브러리 또는 하드코딩된 계수 Convolution 사용.

*   **Min-Max Normalization** (`ApplyMinMax`)
    *   $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$

*   **L2 Normalization** (`ApplyL2`)
    *   $x' = \frac{x}{\sqrt{\sum x_i^2}}$

*   **Mean Centering** (`ApplyCenter`)
    *   학습 시에는 "전체 데이터의 평균"을 뺍니다.
    *   **추론(Runtime) 시**: 단일 라인/픽셀에 대해 이 연산을 수행하면 데이터가 0이 되거나(단일 픽셀) 왜곡될 수 있습니다.
    *   **C# 구현 권장**: 학습된 `Mean Vector`가 제공되지 않는다면 이 옵션은 **사용하지 않는 것(False)**이 안전합니다.
    *   만약 꼭 써야 한다면, 입력된 **Line 전체의 평균**을 사용해야 합니다.

#### 3. Feature Extraction (Dimensionality Reduction)

*   **Simple Derivative (Gap Difference)** (`ApplyDeriv`)
    *   가장 중요하고 빈번하게 사용됨.
    *   파라미터: `Gap`, `DerivOrder` (보통 1)
    *   **Formula**: $D[i] = \text{Band}[i + \text{Gap}] - \text{Band}[i]$
        *   주의: Python 구현(`Right - Left`)과 일치해야 합니다. (Spec v1.0 수정사항)
    *   `DerivOrder > 1`인 경우, 위 연산을 재귀적으로 반복.

*   **3-Point Band Depth** (이름 주의)
    *   파라미터: `Gap`
    *   중심점($C$)과 좌우 어깨($L, R$)를 이용하여 깊이 계산.
    *   $L = \text{Band}[i - \text{Gap}], \quad R = \text{Band}[i + \text{Gap}], \quad C = \text{Band}[i]$
    *   $\text{Baseline} = \frac{L + R}{2}$
    *   $\text{Depth} = 1 - \frac{C}{\text{Baseline}}$

---

### Step 4: Post-Processing (Real-time Blob Analysis)

라인 스캔(Line Scan) 카메라의 특성상 전체 이미지를 기다릴 수 없으므로, **한 줄씩(Line-by-Line) 연결성을 추적**하는 알고리즘을 사용합니다. 이를 통해 노이즈를 제거하고 객체 단위의 정확한 판정을 내립니다.

#### 1. 자료구조 (Active Blob Table)
```csharp
class ActiveBlob {
    public int StartX;      // 객체의 시작 X 좌표 (Min)
    public int EndX;        // 객체의 끝 X 좌표 (Max)
    public int[] Votes;     // 클래스별 투표수 (e.g. [PP: 100, PET: 5])
    public int TotalPixels; // 전체 픽셀 수 (Size)
    public int LastSeenLine;// 마지막으로 관측된 라인 번호 (종료 판정용)
}
List<ActiveBlob> activeBlobs = new List<ActiveBlob>();
```

#### 2. 라인 연결 알고리즘 (Connectivity Logic)
매 라인마다 다음 로직을 수행합니다.

1.  **Run-Length Encoding (RLE)**:
    *   현재 라인의 픽셀 결과를 '덩어리(Segment)'로 묶습니다.
    *   예: `[배경, 배경, PP, PP, PET, 배경]` -> `Seg1(2~3, Class=PP)`, `Seg2(4~4, Class=PET)`
2.  **Overlap Check (연결성 확인)**:
    *   현재 라인의 Segment가 이전 라인의 `ActiveBlob`과 X좌표가 겹치는지 확인합니다.
    *   **겹침 (Overlap)**: 해당 `ActiveBlob`에 픽셀 수와 클래스 투표를 누적(Update)합니다.
    *   **안 겹침 (New)**: 새로운 `ActiveBlob`을 생성하여 리스트에 추가합니다.
3.  **Blob Closing (객체 종료)**:
    *   어떤 `ActiveBlob`이 이번 라인에서 연결되지 않았다면(즉, 물체가 지나감), **종료된 객체**로 간주합니다.
    *   **Majority Voting**: `Votes` 배열에서 가장 표가 많은 클래스를 최종 결과로 선정합니다.
    *   **Eject**: 최종 결과와 위치(X) 정보를 Ejector 시스템으로 전송합니다.
    *   리스트에서 제거(Remove)합니다.

#### 3. 동시 다중 물체 처리 능 (Multi-Object Capability)
이 알고리즘은 **`List<ActiveBlob>`**을 사용하여 여러 물체를 독립적으로 추적합니다.

*   **동시성 (Concurrency)**: 한 라인에 물체가 5개가 있든 10개가 있든, 각각 별도의 `ActiveBlob` 객체로 관리되므로 서로 간섭하지 않습니다.
*   **비동기 판정 (Asynchronous Ejection)**:
    *   물체 A가 끝나면 -> 즉시 A만 판정 및 이젝트.
    *   물체 B는 아직 지나가는 중이라면 -> 계속 추적 (Vote 누적).
    *   따라서 뒷줄에 있는 물체를 기다릴 필요 없이, **각 물체가 끝나는 시점마다 개별적으로** 신호가 나갑니다.

#### 4. 성능 최적화 (Optimization)
*   **Time Complexity**: 한 줄의 픽셀 수(W)보다 Segment 수(N)가 훨씬 적으므로 매우 빠릅니다. ($O(N)$)
*   **Memory**: 전체 이미지를 저장하지 않고, 현재 활성화된 객체 정보만 유지하므로 메모리 사용량이 매우 적습니다.

---

### Step 5: Ejection Control (Physical Mapping)

판정이 완료된 객체에 대해, 물리적인 에어건(Ejector)을 동작시키기 위한 좌표 변환과 타이밍 계산을 수행합니다. 이 연산은 단순 사칙연산이므로 부하가 거의 없습니다 (Zero Latency).

#### 1. Channel Mapping (Spatial)
*   물체의 중심 좌표를 기준으로 담당 솔레노이드 밸브(채널)를 결정합니다.
*   `CenterX = (Blob.StartX + Blob.EndX) / 2`
*   `ChannelID = CenterX / Pixels_Per_Valve`
    *   예: 640px 폭, 64개 밸브라면 `Pixels_Per_Valve = 10`.

#### 2. Dynamic Delay Strategy (Temporal)
물체 길이에 따라 타격 타이밍을 조절하는 **Hybrid Strategy**를 권장합니다.

*   **Case A: 일반 물체 (Normal)**
    *   전략: **Center Hit** (무게중심 타격)
    *   설명: 물체의 꼬리(Tail)가 관측된 후(Blob Close), 물체 중앙이 에어건 위치에 도달할 때까지 지연(Delay)을 줍니다.
    *   `Delay = (Distance_Camera_to_Gun) - (Object_Length / 2)`

*   **Case B: 긴 물체 (Long Object) - Early Trigger**
    *   전략: **Head Hit** (선두 타격)
    *   설명: `Blob.Length > Max_Length_Threshold`인 경우, 물체의 꼬리를 기다리지 않고 **즉시 발사**합니다.
    *   이는 긴 물체의 앞부분이 이미 에어건을 지나치는 것을 방지합니다.

---

## 4. 구현 시 주의사항 (Critical)

1.  **Zero Allocation**: 실시간 처리를 위해 `features` 배열 등은 픽셀마다 `new` 하지 말고, 미리 할당해둔 버퍼를 재사용하십시오.
2.  **Thread Safety**: 여러 라인을 병렬 처리할 경우, `bestClass` 판별 로직(변수)이 스레드 간에 섞이지 않도록 지역 변수만 사용하십시오.
3.  **Boundary Check**: `targetBand + Gap`이 전체 밴드 수(예: 224)를 넘지 않는지 초기화 단계에서 반드시 검증하십시오. (Python에선 이미 검증됨)
4.  **Inverse Masking**: `ProcessingService`와 달리 C# 런타임은 알루미늄 배경을 전제로 하므로 `<` (Less Than) 연산자를 기본으로 지원해야 합니다.

---

## 5. 성능 목표 (Performance Constraints)

*   **Target**: 1 Line (640 pixels) 처리 시간 < **1.0 ms**
*   **Optimization**:
    *   `Pure Raw Mode`: Preprocessing 없이 바로 `Weights` 내적 가능 (가장 빠름)
    *   `SIMD`: `Vector<double>`을 사용하여 내적 연산을 가속화할 것을 권장합니다.
