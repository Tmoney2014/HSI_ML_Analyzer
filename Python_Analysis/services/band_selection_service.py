import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from config import get as cfg_get  # AI가 수정함: 설정 파일 사용

def select_best_bands(data_cube, n_bands=5, method='spa', exclude_indices=None, keep_order=False):
    # AI가 수정함: keep_order 파라미터 추가 - SPA 선택 순서 유지 옵션
    """
    Select top-k significant bands/wavelengths using SPA analysis.
    
    Args:
        data_cube: (H, W, Bands) or (N, Bands) HSI Data
        n_bands: Number of bands to select
        method: 'spa' (Successive Projections Algorithm) or 'variance'
        exclude_indices: List of band indices to ignore (e.g. water absorption)
        
    Returns:
        selected_bands: List[int] Indices of selected bands
        loadings: (Bands,) Importance score (e.g. max projection or variance)
        mean_spectrum: (Bands,) Mean spectrum of data
    """
    
    # Flatten: (N, Bands)
    if data_cube.ndim == 3:
        n_pixels = data_cube.shape[0] * data_cube.shape[1]
        X = data_cube.reshape(n_pixels, -1)
    else:
        X = data_cube
        
    n_features = X.shape[1]
    
    # Handle Exclusion
    valid_indices = np.arange(n_features)
    if exclude_indices is not None:
        mask = np.ones(n_features, dtype=bool)
        mask[exclude_indices] = False
        valid_indices = valid_indices[mask]
        X = X[:, valid_indices]
        
    if X.shape[1] == 0:
        return [], np.zeros(n_features), np.zeros(n_features)

    # AI가 수정함: 데드 코드 제거 (mean_spectrum은 함수 끝에서 global_mean으로 계산됨)
    
    # 1. Selection Logic
    selected_internal_indices = []
    # AI가 수정함: importance_scores를 SPA 루프에서 계산하도록 변경
    importance_scores = np.zeros(X.shape[1])
    
    if method == 'variance':
        # Simple Variance
        importance_scores = np.var(X, axis=0)
        top_k = np.argsort(importance_scores)[::-1][:n_bands]
        selected_internal_indices = top_k.tolist()
        
    else: # Default: SPA-like (using Projections or PCA Loadings)
        # Standard SPA Implementation
        # 1. Downsample if too large (for speed)
        # AI가 수정함: 설정 파일에서 값 로드
        max_samples = cfg_get('spa', 'max_samples', 10000)
        if X.shape[0] > max_samples:
             # Just a safety cap for VERY large data (e.g. >10k) to prevent freezing
             # But use deterministic random
             rng = np.random.RandomState(42)
             indices = rng.choice(X.shape[0], max_samples, replace=False)
             X_spa = X[indices, :]
        else:
             X_spa = X
            
        n_proj = n_bands
        n_col = X_spa.shape[1]
        
        # Selected indices
        # Use the outer variable
        # selected_internal_indices = [] # Already defined line 45
        
        # Current projection matrix (Start with X)
        # We need to project columns (bands). 
        # Standard SPA:
        # 1. Initialize P = X (N x B)
        # 2. Select column j with max norm -> k=0
        # 3. Project P onto orthogonal subspace of selected column
        # 4. Repeat
        
        P = X_spa.copy()
        
        # AI가 수정함: 초기 Norm을 Candidate Bands로 저장 (시각화용)
        initial_norms = np.linalg.norm(P, axis=0)
        importance_scores = initial_norms.copy()
        
        for k in range(n_proj):
            # Calculate norms of all columns (bands)
            norms = np.linalg.norm(P, axis=0)
            
            # Find max norm
            # Mask already selected
            if len(selected_internal_indices) > 0:
                norms[selected_internal_indices] = -1 
            
            max_idx = np.argmax(norms)
            selected_internal_indices.append(max_idx)
            
            if k == n_proj - 1: break
            
            # Project P onto orthogonal space of v_max
            v = P[:, max_idx].reshape(-1, 1) # (N, 1)
            
            # v_norm_sq = v.T @ v
            v_norm_sq = np.dot(v.T, v)
            
            if v_norm_sq == 0: break # Numerical issue
            
            # Projection Operator: proj(y) = (v.T @ y / v.T @ v) * v
            # Matrix form: P_new = P - v @ (v.T @ P / v.T @ v)
            
            factor = np.dot(v.T, P) / v_norm_sq # (1, B)
            projection = np.dot(v, factor)      # (N, B)
            
            P = P - projection
            
    # Map back to original indices
    final_indices = [valid_indices[i] for i in selected_internal_indices]
    
    # AI가 수정함: keep_order=False일 때만 정렬 (기본 동작 유지)
    if not keep_order:
        final_indices.sort()
    
    # Prepare global importance array
    global_importance = np.zeros(n_features)
    global_importance[valid_indices] = importance_scores
    
    # Global Mean
    global_mean = np.zeros(n_features)
    if data_cube.ndim == 3:
        global_mean = np.mean(data_cube.reshape(-1, n_features), axis=0)
    else:
        global_mean = np.mean(data_cube, axis=0)
        
    return final_indices[:n_bands], global_importance, global_mean
