import numpy as np
from sklearn.decomposition import PCA

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
    
    # 4. 중요 밴드 추출
    # pca.components_ 는 (PC개수, 밴드개수) 형태입니다.
    # 각 PC(주성분)를 만들 때, 어떤 밴드가 가장 큰 영향(가중치 절댓값)을 줬는지 찾습니다.
    selected_band_indices = set()
    
    abs_components = np.abs(pca.components_)
    
    for i in range(min(n_bands, abs_components.shape[0])):
        # i번째 주성분에서 가장 기여도가 큰 밴드 인덱스를 찾음
        band_idx = np.argmax(abs_components[i])
        selected_band_indices.add(int(band_idx))
        
    # 5. 혹시 중복된 밴드가 뽑혀서 개수가 모자란 경우 처리
    # (예: PC1과 PC2가 모두 50번 밴드를 가장 중요하다고 뽑을 수 있음)
    # 이럴 때는 전체 기여도 합계가 높은 순서대로 나머지를 채웁니다.
    if len(selected_band_indices) < n_bands:
        flat_loadings = np.sum(abs_components, axis=0)
        sorted_indices = np.argsort(flat_loadings)[::-1]
        for idx in sorted_indices:
            selected_band_indices.add(int(idx))
            if len(selected_band_indices) >= n_bands:
                break
                
    result_list = sorted(list(selected_band_indices))[:n_bands]
    display_list = [x + 1 for x in result_list]
    print(f"   [Band Selection] 최종 선택된 밴드(Display 1-based): {display_list}")
    
    return result_list
