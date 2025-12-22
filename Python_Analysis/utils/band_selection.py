import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

def select_best_bands(data_cube, n_bands=5, method='pca'):
    """
    PCA(주성분 분석)를 사용하여 가장 정보량이 많은(중요한) 밴드를 선택합니다.
    
    [원리 설명]
    초분광 데이터는 밴드 간의 데이터가 매우 비슷합니다(중복성이 높음).
    PCA를 돌리면 데이터의 차이(분산)를 가장 잘 설명하는 '축(PC)'들을 찾습니다.
    이 축을 만드는 데 가장 크게 기여한 원래 밴드가 곧 '중요한 밴드'라고 가정합니다.
    
    Args:
        data_cube: (세로, 가로, 밴드수) 형태의 3차원 데이터
        n_bands: 선택할 밴드의 개수 (기본값: 5개)
        method: 알고리즘 선택 (현재는 'pca'만 구현됨)
        
    Returns:
        selected_band_indices: 선택된 밴드의 인덱스 리스트 (오름차순 정렬됨)
    """
    
    # --- SNR Calculation & Filtering ---
    # 1. 평균 스펙트럼 계산
    mean_spectrum = np.mean(data_cube.reshape(-1, data_cube.shape[-1]), axis=0)
    
    # 2. Savgol Filter로 평활화 (Poly=2, Win=5) - 노이즈 제거된 이상적 신호 추정
    try:
        smoothed = savgol_filter(mean_spectrum, window_length=5, polyorder=2)
        
        # 3. 노이즈 추정 (잔차 = 원본 - 평활화)
        noise_est = np.abs(mean_spectrum - smoothed)
        
        # 4. SNR 계산: 20 * log10(신호 / 노이즈)
        # 신호가 0이거나 노이즈가 0인 경우 방어
        signal_level = np.abs(mean_spectrum)
        snr = 20 * np.log10( (signal_level + 1e-6) / (noise_est + 1e-6) )
        
        # 5. 임계값(Threshold) 미만인 밴드 찾기
        # 예: 30dB 미만이면 노이즈가 심하다고 판단?
        # 근데 절대적 수치는 데이터마다 다름. 상대적으로 하위 5%를 자를까?
        # 여기서는 "값이 너무 튀는" 구간을 잡기 위해, 노이즈가 신호의 1% 이상인 경우 등을 볼 수 있음.
        # 일단 안전하게: SNR이 매우 낮은 하위 밴드들을 제외
        # 여기서는 간단히: "초반 5개" 같은 하드코딩 대신, SNR이 급격히 떨어지는 구간을 찾으면 좋음.
        # 하지만 지금은 시각적 확인이 어려우므로, "노이즈 레벨이 평균보다 매우 높은" 밴드를 찾음.
        
        # 방식: 노이즈 추정치가 전체 평균 노이즈의 3배 이상인 밴드 제거
        mean_noise_level = np.mean(noise_est)
        bad_bands = np.where(noise_est > mean_noise_level * 3.0)[0]
        
        # 추가: 신호 크기 자체가 너무 작은 밴드(0에 가까움)도 제거
        max_signal = np.max(signal_level)
        weak_bands = np.where(signal_level < max_signal * 0.05)[0] # 최대값의 5% 미만
        
        exclude_bands = set(bad_bands) | set(weak_bands)
        
        if exclude_bands:
            display_excludes = sorted([int(b) + 1 for b in exclude_bands])
            print(f"   [Band Selection] Low SNR/Weak Signal Bands Removed: {display_excludes[:10]} ... (Total {len(display_excludes)})")
            
    except Exception as e:
        print(f"   [Band Selection] SNR Calculation Failed: {e}")
        exclude_bands = set()
        
    print(f"   [Band Selection] '{method.upper()}' 알고리즘으로 중요 밴드 {n_bands}개를 찾습니다...")
    
    h, w, b = data_cube.shape
    
    # 1. 2차원 행렬로 변환 (Flatten)
    # PCA는 (샘플 개수, 특징 개수) 형태의 2차원 배열만 입력으로 받습니다.
    # 따라서 이미지(H x W)를 쭉 펴서 샘플로 만들고, 밴드(B)를 특징으로 둡니다.
    n_samples = 10000
    flat_data = data_cube.reshape(-1, b)
    
    # 2. 데이터 샘플링 (속도 향상)
    # 픽셀 전체를 다 쓰면 너무 느리므로, 랜덤하게 10,000개 픽셀만 뽑아서 분석합니다.
    if flat_data.shape[0] > n_samples:
        indices = np.random.choice(flat_data.shape[0], n_samples, replace=False)
        flat_data_subset = flat_data[indices, :]
    else:
        flat_data_subset = flat_data
        
    # 3. PCA 실행
    pca = PCA(n_components=n_bands)
    pca.fit(flat_data_subset)
    
    # 4. 중요 밴드 추출 (Weighted Importance Score + Peak Finding)
    # 4-1. 점수 계산: sum( |Loading_i| * Variance_i )
    abs_components = np.abs(pca.components_)
    explained_variance = pca.explained_variance_ratio_
    weighted_components = abs_components * explained_variance[:, np.newaxis]
    importance_scores = np.sum(weighted_components, axis=0) # (Bands,)
    
    # 4-2. 국소 최댓값(Peak) 찾기 - Iterative Greedy Strategy
    # Scipy find_peaks 대신, "가장 높은 점수"를 무조건 먼저 뽑고
    # 그 주변(distance)을 지워나가는 방식이 사용자의 직관("왜 제일 높은거 안뽑아?")에 가장 부합합니다.
    
    masked_scores = importance_scores.copy()
    selected_band_indices = []
    
    # distance=3: 좌우 3칸씩 (총 7칸 윈도우) 마스킹
    neighbor_dist = 3
    
    # 무한 루프 방지용 카운터
    safety_cnt = 0
    max_iter = len(importance_scores) 
    
    while len(selected_band_indices) < n_bands and safety_cnt < max_iter:
        safety_cnt += 1
        
        # 1. 현재 남은 것 중 1등 찾기
        best_idx = np.argmax(masked_scores)
        best_score = masked_scores[best_idx]
        
        # 만약 남은 점수가 0 이하라면(모두 마스킹됨) 중단
        if best_score <= 0:
            break
            
        # 2. 선택
        if best_idx not in exclude_bands:
            selected_band_indices.append(int(best_idx))
        
        # 3. 주변 마스킹 (자신 포함 좌우 n칸을 0으로 만듦)
        start = max(0, best_idx - neighbor_dist)
        end = min(len(masked_scores), best_idx + neighbor_dist + 1)
        masked_scores[start:end] = -1.0 
    
    result_list = sorted(selected_band_indices)
    display_list = [x + 1 for x in result_list]
    print(f"   [Band Selection] Greedy Selection: {display_list}")
    
    return result_list, importance_scores, mean_spectrum
