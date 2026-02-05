# 🧠 HSI 캐시 아키텍처 (기술 심층 분석)

> **자동 생성됨**: AI Agent 작성  
> **대상 독자**: AI 개발자, 시스템 아키텍트  
> **목적**: 캐싱 로직, 스레드 안전성, 무효화 규칙에 대한 절대적인 기준(Source of Truth) 정의

---

## 1. 시스템 개요 (2단계 계층 구조)

이 시스템은 **메모리 사용량**과 **연산 지연 시간(Latency)** 의 균형을 맞추기 위해 엄격한 **2단계(2-Level) 캐싱 전략**을 채택하고 있습니다.

### 캐시 맵 (Cache Map)

| 레벨 | 컴포넌트 | 변수명 | 범위 (Scope) | 타입 시그니처 (Signature) | 스레드 안전성 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1** | **MainViewModel** | `data_cache` | 전역 (VM 싱글톤) | `SmartCache[str, Tuple[ndarray, list]]` | ✅ **Locked (RLock)** |
| **L2** | **TrainingViewModel** | `cached_base_data` | 세션 (Training/Analysis) | `Dict[str, Tuple[ndarray, None]]` | ⚠️ **메인 스레드 전용 (Main)** |

---

## 2. 타입 정의 (Strict Spec)

Worker와 VM 간의 데이터 일관성을 보장하기 위한 엄격한 타입 정의입니다.

### L1: Raw Data Cache (`SmartCache`)
```python
# Key: 절대 파일 경로 (str)
# Value: (Cube Data, Wavelengths)
ValueType = Tuple[
    np.ndarray,  # Shape: (Height, Width, Bands), Dtype: float32/uint16
    List[float]  # Wavelengths
]
```
*   **불변 규칙 (Invariant)**: 디스크에서 읽은 **RAW DN** 값을 그대로 저장. 마스킹이나 변환 없음.
*   **삭제 정책 (Eviction)**: `max_items` (개수 제한) 및 `min_memory_gb` (RAM 압박 시)에 따라 자동 삭제.

### L2: Base Data Cache (`Dict`)
```python
# Key: 절대 파일 경로 (str)
# Value: (ProcessedData, Reserved)
ValueType = Tuple[
    np.ndarray,  # Shape: (N_Pixels, Bands), Dtype: float32 (유효 픽셀만 존재)
    NoneType     # 예약된 필드 (이전에는 y 라벨이었으나, 현재는 각 Worker가 로컬에서 관리)
]
```
*   **불변 규칙**: **Reflectance/Absorbance** 변환 완료, **Masking** 적용 완료 (무효 픽셀 제거됨).
*   **불변 규칙**: **전처리(Processing) 미적용**. (SG Filter, 미분 등은 적용되지 않음 - Lazy Processing 규칙)

---

## 3. 데이터 흐름 및 스레드 모델

### 3.1. 스레드 권한 규칙 (Thread Affinity)
1.  **L1 Cache (`MainVM`)**: **다중 스레드 접근 가능** (`TrainingWorker`, `OptimizationWorker`, `MainThread`).
    *   반드시 내부적으로 `RLock`을 사용해야 함 (`SmartCache` 구현체 확인).
2.  **L2 Cache (`TrainingVM`)**: **오직 메인 스레드만 접근 가능**.
    *   Worker는 절대로 L2 캐시에 **직접 쓰기(Write) 금지**.
    *   Worker는 `base_data_ready(path, data)` 시그널을 방출(Publish)함.
    *   `TrainingVM` (Main Thread)이 시그널을 수신(Subscribe)하여 Dictionary를 갱신함.
    *   Qt Signal의 직렬화(Serialization) 덕분에 **별도의 Lock 불필요**.

### 3.2. 데이터 인출 파이프라인 (Worker Logic)
```mermaid
graph TD
    A[Worker가 파일 F 요청] -->|L2 확인 (Base)| B{L2 Hit?}
    B -- Yes --> C[Base Data 반환 (초고속)]
    B -- No --> D[L1 확인 (Raw)]
    D -- Hit --> E[Raw Cube 획득]
    D -- Miss --> F[디스크 I/O 로드]
    F --> G[L1 캐시 갱신] --> E
    E --> H[ProcessingService.process_cube]
    H -->|마스킹 + 반사율 변환| I[새로운 Base Data 생성]
    I --> J[base_data_ready 시그널 방출]
    J --> K[TrainingVM이 L2 갱신]
    I --> C
```

---

## 4. 무효화 정책 (Cache Coherency)

데이터의 신뢰성을 위해 다음 규칙에 따라 캐시를 비워야 합니다.

| 이벤트 | 관련 시그널 | L1 동작 | L2 동작 | 트리거 논리 |
| :--- | :--- | :--- | :--- | :--- |
| **반사율 모드 변경** | `mode_changed` | 유지 | **전체 삭제** | 수식이 변경됨 (Raw vs Ref) |
| **White/Dark Ref 변경** | `refs_changed` | 유지 | **전체 삭제** | 분모(기준값)가 변경됨 |
| **Threshold / Mask 변경** | `base_data_invalidated` | 유지 | **전체 삭제** | 유효 픽셀 판정이 변경됨 |
| **파일 제거 (그룹)** | `files_changed` | **키 삭제** | **키 삭제** | 파일이 더 이상 불필요함 |
| **전처리 파라미터 변경** (SG 등) | `model_updated` | 유지 | 유지 | **Lazy Processing** (실시간 재계산) |

> **중요 (Critical)**: Savitzky-Golay, 미분 Gap 등의 **전처리 파라미터 변경은 캐시를 지우지 않습니다.**
> 이는 사용자가 데이터를 다시 로드하거나 마스킹하지 않고도, 빠르게 파라미터를 튜닝할 수 있도록 의도된 설계입니다.

---

## 5. 알려진 함정 및 해결책 (Troubleshooting)

### 🔴 함정 1: 튜플 덮어쓰기 (수정됨)
*   **증상**: `RuntimeError: The truth value of an array ... is ambiguous`.
*   **원인**: 최적화 종료 시 `L2 Dict` 전체를 `Tuple` (X_all, y_all)로 덮어써버림.
*   **해결**: Worker는 오직 `(path, data)` 쌍만 방출하고, `TrainingVM`은 `Dict[path] = data`로 개별 갱신만 수행함. 컨테이너 자체를 교체 금지.

### 🔴 함정 2: 지역 변수 미할당 (수정됨)
*   **원인**: `ProcessingService.get_base_data` 내부 `try-except` 블록 실패 시 `flat_data` 반환값이 정의되지 않음.
*   **해결**: 예외 발생 시 확실하게 `raise` 하거나 변수를 초기화하여 흐름 제어.

---

## 6. 검증 체크리스트 (AI 에이전트용)

시스템을 수정할 때 다음 사항을 반드시 확인하십시오:

- [ ] **Locking**: L1 캐시 수정 시 `SmartCache`의 Lock을 확인했는가?
- [ ] **Thread**: Worker 스레드에서 L2 캐시에 직접 쓰려고 하지 않았는가? (금지됨)
- [ ] **Type**: L2 딕셔너리에 실수로 `Dict` 대신 `Tuple`을 저장하지 않았는가?
- [ ] **Lazy**: 새로운 전처리 단계를 추가할 때 불필요하게 캐시를 무효화하고 있지 않은가? (피해야 함)
