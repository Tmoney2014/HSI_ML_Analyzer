import numpy as np
from scipy.signal import savgol_filter

def create_background_mask(cube, threshold=0.1):
    """
    이미지의 평균 밝기를 기준으로 배경 마스크를 생성합니다.
    
    Args:
        cube: (H, W, Bands) 형태의 3차원 데이터
        threshold: 배경으로 간주할 밝기 임계값 (0.0 ~ 1.0)
                   이 값보다 어두우면 배경(0), 밝으면 물체(1)로 처리
                   
    Returns:
        mask: (H, W) 형태의 2차원 부울(Boolean) 배열
              True(1): 물체, False(0): 배경
    """
    # 밴드 전체의 평균 밝기를 계산 (단순 평균)
    mean_image = np.mean(cube, axis=2)
    
    # 임계값보다 큰 픽셀만 True (물체)
    mask = mean_image > threshold
    
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
