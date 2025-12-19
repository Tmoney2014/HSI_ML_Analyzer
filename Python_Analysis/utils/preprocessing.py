import numpy as np
from scipy.signal import savgol_filter

def create_background_mask(cube, threshold=0.1, band_index=None):
    """
    이미지의 밝기를 기준으로 배경 마스크를 생성합니다. (단일 밴드, 평균, 또는 복합 규칙)
    
    Args:
        cube: (H, W, Bands) 형태의 3차원 데이터
        threshold: 단순 모드에서 사용할 밝기 임계값
        band_index: 
            - None: 전체 평균 사용
            - int: 해당 밴드 인덱스 사용
            - str: 복합 규칙 문자열 (예: "80>0.1; 130<0.3")
                   사용 가능한 연산자: >, <, >=, <=
                   구분자: ; (세미콜론)
                   
    Returns:
        mask: (H, W) 형태의 2차원 부울(Boolean) 배열
    """
    H, W, B = cube.shape
    
    import re

    # 1. 복합 규칙 문자열 처리 (Advanced String Rules with & |)
    if isinstance(band_index, str) and any(op in band_index for op in ['>', '<']):
        try:
            # 정규표현식으로 "b숫자 연산자 숫자" 패턴 찾기 (예: b80 > 0.1)
            # [bB]? : 옵션 'b' 접두사
            # (\d+) : 밴드 인덱스 (Group 1)
            # ([><=]+) : 연산자 (Group 2)
            # ([\d.]+(?:[eE][+-]?\d+)?) : 실수 값 (Group 3)
            pattern = r'[bB]?(\d+)\s*([><=]+)\s*([\d.]+(?:[eE][+-]?\d+)?)'
            
            # 모든 조건을 찾아서 미리 마스크 계산
            matches = list(re.finditer(pattern, band_index))
            mask_dict = {}
            processed_rule = band_index
            
            for i, match in enumerate(matches):
                full_expr = match.group(0)
                idx = int(match.group(1))
                op = match.group(2)
                val = float(match.group(3))
                
                # 마스크 키 생성 (eval에서 사용할 변수명)
                key = f"MASK_{i}"
                
                # 마스크 계산
                if 0 <= idx < B:
                    band_img = cube[:, :, idx]
                    if op == '>=': sub_mask = band_img >= val
                    elif op == '<=': sub_mask = band_img <= val
                    elif op == '>': sub_mask = band_img > val
                    elif op == '<': sub_mask = band_img < val
                    else: sub_mask = np.zeros((H, W), dtype=bool) # Fallback
                else:
                    sub_mask = np.zeros((H, W), dtype=bool) # Invalid Band
                
                mask_dict[key] = sub_mask
                
                # 원본 문자열에서 해당 조건식을 변수명으로 치환
                # 단순 replace는 중복 발생 시 위험하므로, span을 이용하거나 주의 필요.
                # 여기서는 match된 정확한 부분 문자열을 replace (가장 안전하게는 역순 치환이 좋음)
                pass 
            
            # 안전한 치환을 위해 역순으로 처리
            matches.reverse()
            for i, match in enumerate(matches):
                # i는 역순이므로 원래 인덱스는 len-1-i
                original_idx = len(matches) - 1 - i
                key = f"MASK_{original_idx}"
                start, end = match.span()
                processed_rule = processed_rule[:start] + key + processed_rule[end:]
                
            # eval 사용하여 논리 연산 수행 (&, |)
            # 예: "MASK_0 & MASK_1 | MASK_2"
            final_mask = eval(processed_rule, {"__builtins__": None}, mask_dict)
            
            return final_mask.astype(bool)

        except Exception as e:
            print(f"Mask Rule Parse Error: {e}")
            # 에러 발생 시 평균 기준 Fallback
            criterion_image = np.mean(cube, axis=2)
            mask = criterion_image > threshold
            return mask

    # 2. 단일 밴드 인덱스 (Single Band Mode)
    elif isinstance(band_index, int) or (isinstance(band_index, str) and band_index.isdigit()):
        idx = int(band_index)
        if 0 <= idx < B:
            criterion_image = cube[:, :, idx]
        else:
            criterion_image = np.mean(cube, axis=2)
            
    # 3. 기본값 (Default: Mean Intensity)
    else:
        criterion_image = np.mean(cube, axis=2)
    
    # 임계값 적용
    mask = criterion_image > threshold
    return mask

def apply_mask(cube, mask):
    """
    마스크를 적용하여 배경 픽셀을 제거하고, 1줄(Flatten)로 폅니다.
    
    Args:
        cube: (H, W, Bands) 원본 데이터
        mask: (H, W) 2차원 마스크 (True: 물체)
        
    Returns:
        masked_data: (N_pixels, Bands) 형태의 2차원 배열
                     배경이 제거된 픽셀들만 모여 있음
    """
    # 마스크가 True인 위치의 픽셀들만 추출
    masked_data = cube[mask]
    
    return masked_data

def apply_snv(data):
    """
    SNV (Standard Normal Variate) 보정을 적용합니다.
    스펙트럼의 산란(Scattering) 효과나 조명 변화를 줄여줍니다.
    각 픽셀(스펙트럼)마다 평균을 0, 표준편차를 1로 만듭니다.
    
    Args:
        data: (N_pixels, Bands) 형태의 2차원 배열
        
    Returns:
        snv_data: SNV 보정된 데이터
    """
    # 각 행(픽셀)별로 평균과 표준편차 계산
    # keepdims=True를 해야 (N, 1) 형태가 되어 방송(Broadcasting) 연산이 됨
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    
    # 표준편차가 0이면 나눗셈 에러가 나므로 살짝 더해줌
    std[std == 0] = 1e-10
    
    snv_data = (data - mean) / std
    return snv_data

def apply_savgol(data, window_size=5, poly_order=2, deriv=0):
    """
    Savitzky-Golay 필터를 적용하여 노이즈를 제거하거나 미분을 수행합니다.
    
    Args:
        data: (N_pixels, Bands) 입력 데이터
        window_size: 필터 윈도우 크기 (홀수여야 함, 예: 5, 7, 11)
        poly_order: 다항식 차수 (보통 2 또는 3)
        deriv: 미분 차수 (0: 스무딩만, 1: 1차 미분, 2: 2차 미분)
        
    Returns:
        filtered_data: 필터링된 데이터
    """
    # scipy 라이브러리의 savgol_filter 사용
    # axis=1 (밴드 방향)으로 필터 적용
    filtered_data = savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv, axis=1)
    
    return filtered_data

def apply_mean_centering(data):
    """
    평균 중심화 (Mean Centering)
    각 밴드의 평균값을 0으로 맞춥니다.
    
    Args:
        data: (N_pixels, Bands)
        
    Returns:
        centered_data
    """
    mean_spectrum = np.mean(data, axis=0)
    centered_data = data - mean_spectrum
    return centered_data

def apply_l2_norm(data):
    """
    L2 정규화 (Vector Normalization)
    각 스펙트럼 벡터의 길이를 1로 만듭니다.
    조명 밝기 변화에 매우 강인합니다.
    식: X_new = X / sqrt(sum(X^2))
    
    Args:
        data: (N_pixels, Bands)
        
    Returns:
        norm_data
    """
    # 각 행(픽셀)의 L2 Norm(유클리드 거리) 계산
    l2_norms = np.linalg.norm(data, axis=1, keepdims=True)
    
    # 0으로 나누기 방지
    l2_norms[l2_norms == 0] = 1e-10
    
    norm_data = data / l2_norms
    return norm_data

def apply_minmax_norm(data):
    """
    Min-Max 정규화
    데이터 범위를 0~1 사이로 맞춥니다.
    식: X_new = (X - min) / (max - min)
    
    Args:
        data: (N_pixels, Bands)
        
    Returns:
        norm_data
    """
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-10
    
    norm_data = (data - min_vals) / range_vals
    return norm_data
