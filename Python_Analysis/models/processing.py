import numpy as np
from scipy.signal import savgol_filter
import re

def create_background_mask(cube, threshold=0.1, band_index=None):
    """
    Generate background mask based on intensity or composite rules.
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        # AI가 추가함: threshold 파싱 실패 경고 — silent fallback 대신 명시적 경고
        print(f"Warning: [create_background_mask] Invalid threshold value '{threshold}' — using 0.0 (all pixels included)")
        threshold = 0.0
        
    H, W, B = cube.shape
    
    # 1. Complex Rules (Advanced String Rules with & |)
    # AI가 수정함: != / == 연산자도 complex rule 경로로 진입하도록 감지 조건 확장
    if isinstance(band_index, str) and any(op in band_index for op in ['>', '<', '==', '!=']):
        try:
            # AI가 수정함: != 연산자 지원 추가 — [><=!]+ 로 패턴 확장
            pattern = r'[bB]?(\d+)\s*([><=!]+)\s*([\d.]+(?:[eE][+-]?\d+)?)'
            matches = list(re.finditer(pattern, band_index))
            mask_dict = {}
            processed_rule = band_index
            
            for i, match in enumerate(matches):
                idx = int(match.group(1))
                op = match.group(2)
                val = float(match.group(3))
                key = f"MASK_{i}"
                
                if 0 <= idx < B:
                    band_img = cube[:, :, idx]
                    if op == '>=': sub_mask = band_img >= val
                    elif op == '<=': sub_mask = band_img <= val
                    elif op == '>': sub_mask = band_img > val
                    elif op == '<': sub_mask = band_img < val
                    # AI가 추가함: == / != 연산자 지원 — C# MaskRule 패리티 (동시 추가)
                    elif op == '==': sub_mask = band_img == val
                    elif op == '!=': sub_mask = band_img != val
                    else:
                        print(f"Warning: [create_background_mask] Unsupported operator '{op}' — rule skipped (returns empty mask)")
                        sub_mask = np.zeros((H, W), dtype=bool)
                else:
                    sub_mask = np.zeros((H, W), dtype=bool)
                
                mask_dict[key] = sub_mask
                
            matches.reverse()
            for i, match in enumerate(matches):
                original_idx = len(matches) - 1 - i
                key = f"MASK_{original_idx}"
                start, end = match.span()
                processed_rule = processed_rule[:start] + key + processed_rule[end:]
                
            final_mask = eval(processed_rule, {"__builtins__": None}, mask_dict)
            return final_mask.astype(bool)

        except Exception as e:
            print(f"Mask Rule Parse Error: {e}")
            criterion_image = np.mean(cube, axis=2)
            mask = criterion_image > threshold
            return mask

    # 2. Single Band
    elif isinstance(band_index, int) or (isinstance(band_index, str) and band_index.isdigit()):
        idx = int(band_index)
        if 0 <= idx < B:
            criterion_image = cube[:, :, idx]
        else:
            criterion_image = np.mean(cube, axis=2)
            
    # 3. Default (Mean)
    else:
        criterion_image = np.mean(cube, axis=2)
    
    mask = criterion_image > threshold
    return mask

def apply_mask(cube, mask):
    return cube[mask]

def apply_snv(data):
    # AI가 수정함: zero-std guard — C# SnvProcessor 패리티 일치 (ddof=1, threshold 1e-9)
    # C#: if (std > 1e-9) 일 때만 정규화, 그 외 원본 유지
    # 이전 방식 (std==0 → 1e-10)은 수치 폭발 유발 + 수치 근-0 미처리 문제
    # 학술 표준 (Barnes et al., 1989): 표본 표준편차 (N-1, ddof=1) 사용
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, ddof=1, keepdims=True)
    result = data.copy()
    valid = std.squeeze(axis=1) > 1e-9
    result[valid] = (data[valid] - mean[valid]) / std[valid]
    return result

def apply_savgol(data, window_size=5, poly_order=2, deriv=0):
    # AI가 추가함: 입력 검증 — C# SavitzkyGolayProcessor 패리티 일치
    # C#: 짝수 windowSize → 홀수로 자동 올림. Python도 동일 처리.
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= poly_order:
        raise ValueError(
            f"Strict Mode Error: window_size({window_size}) must be > poly_order({poly_order}). "
            "SG filter 요구사항: window_length > polyorder."
        )
    if data.shape[1] < window_size:
        raise ValueError(
            f"Strict Mode Error: Not enough bands for SG filter. "
            f"Has {data.shape[1]}, Need >= {window_size}."
        )
    return savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv, axis=1)

def apply_min_subtraction(data):
    """
    Subtract the minimum value from each spectrum (Baseline Correction).
    Shifts the spectrum so the lowest point is at 0.
    """
    min_vals = np.min(data, axis=1, keepdims=True)
    return data - min_vals

def apply_l2_norm(data):
    # AI가 수정함: zero-norm guard — C# L2NormalizeProcessor 패리티 일치
    # C#: if (sumSq > 1e-9) 일 때만 정규화, 그 외 원본 유지
    # 이전 방식 (l2_norms==0 → 1e-10)은 수치 폭발(÷1e-10) 유발
    l2_norms = np.linalg.norm(data, axis=1, keepdims=True)
    result = data.copy()
    valid = l2_norms.squeeze(axis=1) > np.sqrt(1e-9)  # ~3.16e-5 임계값 (C# sumSq>1e-9 동등)
    result[valid] = data[valid] / l2_norms[valid]
    return result

def apply_minmax_norm(data):
    # AI가 수정함: zero-range guard — C# MinMaxProcessor 패리티 일치
    # C#: if (range > 1e-9) 일 때만 정규화, 그 외 원본 유지
    # 이전 방식 (range==0 → 1e-10)은 수치 폭발(÷1e-10) 유발
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    result = data.copy()
    valid = range_vals.squeeze(axis=1) > 1e-9
    result[valid] = (data[valid] - min_vals[valid]) / range_vals[valid]
    return result

def apply_simple_derivative(data, gap: int = 5, order: int = 1):
    """
    /// <ai>AI가 작성함</ai>
    Gap Difference (Simple Differentiation).
    주어진 Gap 만큼 떨어진 밴드 간의 차이를 계산합니다.
    공식: Output[i] = Band[i + gap] - Band[i]  (Forward Difference, numpy.diff 방향)
    
    표준 방향: Forward Difference (학계 표준 — numpy.diff, R prospectr::gapDer 동일)
    C# 런타임 RawGapFeatureExtractor / LogGapFeatureExtractor 동일 방향.
    
    이 과정을 거치면 데이터의 밴드 수가 (Gap * order)만큼 줄어듭니다.
    
    Args:
        data: (N_pixels, Bands) 입력 데이터
        gap: 밴드 간 간격 (Default: 5)
        order: 미분 반복 횟수 (Default: 1)
        
    Returns:
        diff_data: (N_pixels, New_Bands) 차분된 데이터
    """
    if gap < 1: return data
    diff_data = data.copy()

    for i in range(order):
        # AI가 수정함: 밴드 부족 시 에러 발생 (학습 기만 방지)
        if diff_data.shape[1] <= gap:
            raise ValueError(f"Strict Mode Error: Not enough bands for Derivative order {i+1}. Has {diff_data.shape[1]}, Need > {gap}.")

        # Forward Difference: Band[i+gap] - Band[i]
        # A = data[:, :-gap] (Index 0 ~ N-Gap) — Left (Current)
        # B = data[:, gap:]  (Index Gap ~ N)   — Right (Future)
        A = diff_data[:, :-gap]
        B = diff_data[:, gap:]
        diff_data = B - A

    return diff_data

def apply_absorbance(data, epsilon=1e-6):
    """
    /// <ai>AI가 작성함</ai>
    Apply Pseudo-Absorbance Transformation.
    Formula: A = -log10(R)
    
    Args:
        data: (N_pixels, Bands) - usually Reflectance (0~1)
        epsilon: Safety margin to avoid log(0)
        
    Returns:
        absorbance_data: (N_pixels, Bands)
    """
    # 1. Safety Clipping (0 이하, 음수 방지)
    # 데이터가 0이면 log가 무한대로 발산하므로 아주 작은 값으로 대체
    local_R = np.maximum(data, epsilon)
    
    # 2. Log Transformation
    # Reflectance가 1.0보다 크면 음수가 나올 수 있는데, 보통 흡광도는 양수 취급
    # 하지만 물리적으로 형광이 아니면 R<=1.0 이므로, 그냥 로그 취함.
    absorbance = -np.log10(local_R)
    
    return absorbance
