import numpy as np
from scipy.signal import savgol_filter
import re

def create_background_mask(cube, threshold=0.1, band_index=None):
    """
    Generate background mask based on intensity or composite rules.
    """
    try:
        threshold = float(threshold)
    except:
        threshold = 0.0
        
    H, W, B = cube.shape
    
    # 1. Complex Rules (Advanced String Rules with & |)
    if isinstance(band_index, str) and any(op in band_index for op in ['>', '<']):
        try:
            pattern = r'[bB]?(\d+)\s*([><=]+)\s*([\d.]+(?:[eE][+-]?\d+)?)'
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
                    else: sub_mask = np.zeros((H, W), dtype=bool) 
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
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std[std == 0] = 1e-10
    return (data - mean) / std

def apply_savgol(data, window_size=5, poly_order=2, deriv=0):
    return savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv, axis=1)

def apply_mean_centering(data):
    mean_spectrum = np.mean(data, axis=0)
    return data - mean_spectrum

def apply_min_subtraction(data):
    """
    Subtract the minimum value from each spectrum (Baseline Correction).
    Shifts the spectrum so the lowest point is at 0.
    """
    min_vals = np.min(data, axis=1, keepdims=True)
    return data - min_vals

def apply_l2_norm(data):
    l2_norms = np.linalg.norm(data, axis=1, keepdims=True)
    l2_norms[l2_norms == 0] = 1e-10
    return data / l2_norms

def apply_minmax_norm(data):
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-10
    return (data - min_vals) / range_vals

def apply_simple_derivative(data, gap: int = 5, order: int = 1, apply_ratio: bool = False, ndi_threshold: float = 1e-4):
    """
    /// <ai>AI가 작성함</ai>
    Gap Difference (Simple Differentiation).
    주어진 Gap 만큼 떨어진 밴드 간의 차이를 계산합니다.
    공식: Output[i] = Band[i] - Band[i + gap]
    
    [Option] Apply Ratio (NDI):
    공식: Output[i] = (Band[i] - Band[i + gap]) / (Band[i] + Band[i + gap] + epsilon)
    
    이 과정을 거치면 데이터의 밴드 수가 (Gap * order)만큼 줄어듭니다.
    
    Args:
        data: (N_pixels, Bands) 입력 데이터
        gap: 밴드 간 간격 (Default: 5)
        order: 미분 반복 횟수 (Default: 1)
        apply_ratio: Band Ratio (NDI) 적용 여부 (Default: False)
        
    Returns:
        diff_data: (N_pixels, New_Bands) 차분된 데이터
    """
    if gap < 1: return data
    if order < 1: 
        raise ValueError(f"Strict Mode: Derivative order must be >= 1 (Got {order})")
    
    
    diff_data = data.copy()
    epsilon = 1e-6
    
    for i in range(order):
        # AI가 수정함: 밴드 부족 시 에러 발생 (학습 기만 방지)
        if diff_data.shape[1] <= gap:
            raise ValueError(f"Strict Mode Error: Not enough bands for Derivative order {i+1}. Has {diff_data.shape[1]}, Need > {gap}.")
        
        # User Logic: data[:, :, :-GAP] - data[:, :, GAP:]
        # Note: 
        # A = data[:, :-gap] (Index 0 ~ N-Gap)
        # B = data[:, gap:]  (Index Gap ~ N)
        
        # Correct Logic: Band[i+gap] - Band[i] (Right - Left)
        # This matches standard derivative definition (Delta y)
        if diff_data.shape[1] > gap:
            A = diff_data[:, :-gap] # Left (Current)
            B = diff_data[:, gap:]  # Right (Future)
            
            if apply_ratio:
                # NDI Formula: (B - A) / (B + A)
                # Note: NDI is typically (NIR - RED) / (NIR + RED)
                # Here we strictly follow (Target - Base) / (Target + Base)
                numerator = B - A
                denominator = B + A
                
                mask_valid = denominator > ndi_threshold
                diff_slice = np.zeros_like(numerator)
                diff_slice[mask_valid] = numerator[mask_valid] / (denominator[mask_valid] + epsilon)
                diff_data = diff_slice
            else:
                # Standard Derivative: B - A
                diff_data = B - A
        
    return diff_data

def apply_rolling_3point_depth(data, gap: int = 5):
    """
    /// <ai>AI가 작성함</ai>
    Rolling 3-Point Band Depth (Continuum Removal Lite).
    
    Formula:
        L = Band[i - gap]
        R = Band[i + gap]
        C = Band[i]
        
        Baseline = (L + R) / 2
        Depth = 1 - (C / Baseline)
        
    Result:
        1.0: Center is 0 (Perfect Absorption)
        0.0: Center == Baseline (Flat)
        < 0: Center > Baseline (Peak, not Valley)
        
    Args:
        data: (N_pixels, Bands)
        gap: Distance to shoulders (default: 5)
        
    Returns:
        depth_data: (N_pixels, New_Bands)
        New band count = Original - (2 * gap)
    """
    if gap < 1: 
         raise ValueError(f"Strict Mode: Gap must be >= 1 (Got {gap})")
    
    # Needs at least (2*gap + 1) bands
    if data.shape[1] < (2 * gap + 1):
        # AI가 수정함: 밴드 부족 시 에러 발생
        raise ValueError(f"Strict Mode Error: Not enough bands for 3-Point Depth. Has {data.shape[1]}, Need {2*gap+1}.")
        
    L = data[:, :-2*gap]      # Left Shoulder
    C = data[:, gap:-gap]     # Center
    R = data[:, 2*gap:]       # Right Shoulder
    
    # Baseline (Average of Shoulders)
    baseline = (L + R) / 2.0
    
    # Avoid division by zero
    epsilon = 1e-6
    mask_valid = baseline > 1e-5
    
    depth_data = np.zeros_like(C)
    
    # Calculate Depth: 1 - (C / Baseline)
    # Using User's logic: Score = 1 - (2*C / (L+R))
    # Same as: 1 - (C / ((L+R)/2))
    
    depth_data[mask_valid] = 1.0 - (C[mask_valid] / (baseline[mask_valid] + epsilon))
    
    return depth_data

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
