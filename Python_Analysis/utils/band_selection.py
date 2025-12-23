import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

def select_best_bands(data_cube, n_bands=5, method='spa', exclude_indices=None):
    """
    중요 밴드를 선택합니다. (Default: SPA)
    
    Args:
        data_cube: (세로, 가로, 밴드수) 형태의 3차원 데이터
        n_bands: 선택할 밴드의 개수 (기본값: 5개)
        method: 'spa' 또는 'pca'
        exclude_indices: 제외할 밴드 인덱스 리스트 (list of int, 0-based)
    """
    
    # 1. 전처리: Flatten & Subsampling
    h, w, b = data_cube.shape
    flat_data = data_cube.reshape(-1, b)
    
    # 속도 향상을 위한 샘플링 (최대 5000픽셀)
    n_samples = 5000
    if flat_data.shape[0] > n_samples:
        indices = np.random.choice(flat_data.shape[0], n_samples, replace=False)
        X = flat_data[indices, :]
    else:
        X = flat_data
        
    mean_spectrum = np.mean(X, axis=0)
    scores = np.zeros(b) # 시각화용 점수 (SPA는 투영 크기, PCA는 Loading)

    # 안전하게 Set으로 변환
    if exclude_indices is None: exclude_indices = set()
    else: exclude_indices = set(exclude_indices)

    # ==========================================
    # METHOD 1: SPA (Successive Projections Algorithm)
    # ==========================================
    if method.lower() == 'spa':
        print(f"   [Band Selection] Running SPA... (Excluded: {len(exclude_indices)} bands)")
        
        selected_indices = []
        
        # 1. 첫 번째 밴드 선택 (Norm 기준)
        norms = np.linalg.norm(X, axis=0)
        
        # 제외 밴드의 Norm을 -1로 설정하여 선택 방지
        for ex_idx in exclude_indices:
            if 0 <= ex_idx < b:
                norms[ex_idx] = -1.0
                
        first_band = np.argmax(norms)
        if norms[first_band] <= 0:
            print("   [SPA Error] All bands excluded or zero signal.")
            return [], scores, mean_spectrum
            
        selected_indices.append(first_band)
        scores[first_band] = norms[first_band]
        
        # 현재까지 선택된 밴드들의 Orthogonal Subspace에 투영하면서 진행
        current_X = X.copy()
        
        for i in range(n_bands - 1):
            last_selected_idx = selected_indices[-1]
            u = current_X[:, last_selected_idx]
            
            denom = np.dot(u, u)
            if denom < 1e-10: denom = 1e-10
            
            projections = np.dot(u, current_X)
            factor = projections / denom
            subtraction = np.outer(u, factor)
            current_X = current_X - subtraction
            
            # 다음 밴드 찾기
            current_norms = np.linalg.norm(current_X, axis=0)
            
            # 이미 선택된 것 + 제외 목록 Masking
            for selected in selected_indices:
                current_norms[selected] = -1.0
            for ex in exclude_indices:
                if 0 <= ex < b:
                    current_norms[ex] = -1.0
                
            next_band = np.argmax(current_norms)
            max_norm_val = current_norms[next_band]
            
            if max_norm_val <= 0:
                print("   [SPA] 더 이상 유의미한 밴드가 없습니다.")
                break
                
            selected_indices.append(next_band)
            scores[next_band] = max_norm_val
            
        print(f"   [SPA] Selected: {[x+1 for x in selected_indices]}")
        
        result_list = sorted(selected_indices)
        if np.max(scores) > 0: scores = scores / np.max(scores)
        return result_list, scores, mean_spectrum

    # ==========================================
    # METHOD 2: PCA (Fallback)
    # ==========================================
    else:
        # 기존 PCA 로직 유지
        pca = PCA(n_components=n_bands)
        pca.fit(X)
        
        abs_components = np.abs(pca.components_)
        flat_loadings = np.sum(abs_components, axis=0)
        
        selected_band_indices = set()
        for i in range(min(n_bands, abs_components.shape[0])):
            sorted_indices = np.argsort(abs_components[i])[::-1]
            
            # 제외 목록에 없는 1등 찾기
            for rank_idx in sorted_indices:
                if int(rank_idx) not in exclude_indices:
                    selected_band_indices.add(int(rank_idx))
                    break
            
        # Fill rest
        if len(selected_band_indices) < n_bands:
            sorted_total = np.argsort(flat_loadings)[::-1]
            for idx in sorted_total:
                if int(idx) not in exclude_indices and int(idx) not in selected_band_indices:
                    selected_band_indices.add(int(idx))
                    if len(selected_band_indices) >= n_bands: break
                
        result_list = sorted(list(selected_band_indices))[:n_bands]
        
        if np.max(flat_loadings) > 0: flat_loadings /= np.max(flat_loadings)
        return result_list, flat_loadings, mean_spectrum
