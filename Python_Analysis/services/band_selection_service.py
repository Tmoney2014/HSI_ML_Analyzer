import numpy as np
import warnings  # AI가 수정함: Task 2 - UserWarning 출력용
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # AI가 수정함: Task 2 - LDA 기반 방법 추가
from sklearn.feature_selection import f_classif  # AI가 수정함: ANOVA-F 방법 추가
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit  # AI가 수정함: Task 2 - SPA-LDA greedy용
from scipy.signal import savgol_filter
from config import get as cfg_get  # AI가 수정함: 설정 파일 사용

_SUPERVISED_METHODS = ('anova_f', 'spa_lda_fast', 'spa_lda_greedy', 'lda_coef')  # AI가 수정함: supervised 방법 목록 정의


def _stratified_subsample(X, labels, max_samples, random_state=42):
    """클래스 비율을 유지한 서브샘플링. Stratified 불가 시 random fallback."""  # AI가 수정함: stratified 서브샘플링 헬퍼 추가
    if X.shape[0] <= max_samples:
        return X, labels
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=random_state)
        sub_idx, _ = next(sss.split(X, labels))
    except ValueError:
        # 클래스 수 부족 또는 샘플 수 제약으로 stratified 불가 → random fallback
        rng = np.random.RandomState(random_state)
        sub_idx = rng.choice(X.shape[0], max_samples, replace=False)
    return X[sub_idx], labels[sub_idx]


def select_best_bands(data_cube, n_bands=5, method='spa', exclude_indices=None, keep_order=False, labels=None):  # AI가 수정함: labels 인자 추가
    # AI가 수정함: keep_order 파라미터 추가 - SPA 선택 순서 유지 옵션
    """
    Select top-k significant bands/wavelengths using SPA analysis.
    
    Args:
        data_cube: (H, W, Bands) or (N, Bands) HSI Data
        n_bands: Number of bands to select
        method: 'spa' (Successive Projections Algorithm), 'full' (use all valid bands), or supervised methods<!-- AI가 수정함: method 설명 확장 -->
        exclude_indices: List of band indices to ignore (e.g. water absorption)  # AI가 수정함: 기존 인자 설명 유지
        labels: Supervised methods용 class labels (anova_f, spa_lda_fast, spa_lda_greedy, lda_coef)  # AI가 수정함: labels 설명 추가
        
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
    if method not in ('spa', 'full') + _SUPERVISED_METHODS:  # AI가 수정함: supervised 방법 허용 및 unknown method 검증
        raise ValueError(f"Unknown band selection method: {method!r}. Must be one of 'spa', 'full', 'anova_f', 'spa_lda_fast', 'spa_lda_greedy', 'lda_coef'.")  # AI가 수정함: 허용 method 메시지 갱신
    # AI가 수정함: supervised 방법은 labels 필수
    if method in _SUPERVISED_METHODS and labels is None:  # AI가 수정함: labels 누락 방지
        raise ValueError(f"method='{method}' requires labels parameter (got None).")  # AI가 수정함: labels 필수 오류
    
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
    
    if method == 'full':  # AI가 수정함: Variance → Full Band (모든 유효 밴드 사용)
        # Full Band: 모든 valid indices를 그대로 사용 (밴드 선택 알고리즘 없음)  # AI가 수정함:
        importance_scores = np.var(X, axis=0)  # AI가 수정함: importance 시각화용 (band_importance.png)
        selected_internal_indices = list(range(X.shape[1]))  # AI가 수정함:
        
    elif method == 'anova_f':  # AI가 수정함: ANOVA-F supervised 방법
        # 단일 클래스 가드
        assert labels is not None  # AI가 수정함: Pyright 타입 좁히기 및 labels 비어있음 방지
        if len(np.unique(labels)) < 2:  # AI가 수정함: 최소 2개 클래스 확인
            raise ValueError("anova_f: 클래스가 2개 이상 필요합니다.")  # AI가 수정함: 클래스 수 오류
        # labels를 X와 동일한 유효 밴드 슬라이싱 이후에 사용 (이미 exclude 처리됨)
        f_scores, _ = f_classif(X, labels)  # AI가 수정함: ANOVA F-통계량 계산
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0)  # AI가 수정함: 상수 밴드 NaN/Inf 방어
        importance_scores = f_scores  # AI가 수정함: importance score를 F-통계량으로 사용
        selected_internal_indices = list(np.argsort(f_scores)[::-1][:n_bands])  # AI가 수정함: 상위 밴드 선택
        
    elif method == 'lda_coef':  # AI가 수정함: LDA 계수 기반 밴드 중요도
        assert labels is not None
        # subsample (labels와 X 반드시 동일 인덱스)
        max_samples = cfg_get('spa', 'max_samples', 10000)
        if not isinstance(max_samples, int):
            max_samples = 10000
        X_lda, labels_lda = _stratified_subsample(X, labels, max_samples)  # AI가 수정함: stratified 서브샘플링으로 교체
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')  # AI가 수정함: solver='svd' 절대 금지
        try:
            lda.fit(X_lda, labels_lda)
            coef_importance = np.sum(np.abs(lda.coef_), axis=0)
        except Exception:
            warnings.warn("lda_coef fit failed, falling back to variance", RuntimeWarning)
            coef_importance = np.var(X_lda, axis=0)
        importance_scores = coef_importance
        selected_internal_indices = list(np.argsort(coef_importance)[::-1][:n_bands])

    elif method == 'spa_lda_fast':  # AI가 수정함: SPA 후보 추출 → LDA scoring
        assert labels is not None
        # subsample
        max_samples = cfg_get('spa', 'max_samples', 10000)
        if not isinstance(max_samples, int):
            max_samples = 10000
        X_s, labels_s = _stratified_subsample(X, labels, max_samples)  # AI가 수정함: stratified 서브샘플링으로 교체
        # Step 1: SPA로 n_candidates개 후보 추출
        n_candidates = min(X_s.shape[1], max(n_bands * 3, n_bands + 2))  # AI가 수정함: 후보 풀 크기
        P = X_s.copy()
        spa_candidates = []
        for k in range(n_candidates):
            norms = np.linalg.norm(P, axis=0)
            if spa_candidates:
                norms[spa_candidates] = -1
            max_idx = int(np.argmax(norms))
            spa_candidates.append(max_idx)
            if k == n_candidates - 1:
                break
            v = P[:, max_idx].reshape(-1, 1)
            v_norm_sq = float(np.dot(v.T, v))
            if v_norm_sq < 1e-12:
                break
            P = P - np.dot(v, np.dot(v.T, P) / v_norm_sq)
        # Step 2: 후보 중 LDA coef 기반 scoring
        X_cand = X_s[:, spa_candidates]
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')  # AI가 수정함: solver='svd' 금지
        try:
            lda.fit(X_cand, labels_s)
            coef_cand = np.sum(np.abs(lda.coef_), axis=0)
        except Exception:
            coef_cand = np.var(X_cand, axis=0)
        top_local = list(np.argsort(coef_cand)[::-1][:n_bands])
        selected_internal_indices = [spa_candidates[i] for i in top_local]
        importance_scores[spa_candidates] = coef_cand  # AI가 수정함: 모든 SPA candidate LDA 점수 저장 (시각화용)

    elif method == 'spa_lda_greedy':  # AI가 수정함: Greedy cross-validation 밴드 선택
        assert labels is not None
        warnings.warn(  # AI가 수정함: 느린 방법 경고
            "spa_lda_greedy is slow; not recommended for optimization loops.",
            UserWarning, stacklevel=3
        )
        # subsample
        max_samples = cfg_get('spa', 'max_samples', 10000)
        if not isinstance(max_samples, int):
            max_samples = 10000
        X_g, labels_g = _stratified_subsample(X, labels, max_samples)  # AI가 수정함: stratified 서브샘플링으로 교체
        # cv = min(3, min_class_count) — 필수 제약
        class_counts = np.bincount(labels_g.astype(int))
        min_class_count = int(np.min(class_counts[class_counts > 0]))
        cv = min(3, min_class_count)  # AI가 수정함: cv 제한 필수

        remaining = list(range(X_g.shape[1]))
        greedy_selected = []
        greedy_scores = np.zeros(X_g.shape[1])

        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')  # AI가 수정함: solver='svd' 금지
        for _ in range(n_bands):
            if not remaining:
                break
            best_band = None
            best_score = -1.0
            for b in remaining:
                candidate_set = greedy_selected + [b]
                X_trial = X_g[:, candidate_set]
                try:
                    scores = cross_val_score(lda, X_trial, labels_g, cv=cv, scoring='accuracy')
                    score = float(np.mean(scores))
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_band = b
            if best_band is None:
                break
            greedy_selected.append(best_band)
            greedy_scores[best_band] = best_score
            remaining.remove(best_band)

        selected_internal_indices = greedy_selected
        importance_scores = greedy_scores
        
    else: # Default: SPA-like (using Projections or PCA Loadings)
        # Standard SPA Implementation
        # 1. Downsample if too large (for speed)
        # AI가 수정함: 설정 파일에서 값 로드
        max_samples = cfg_get('spa', 'max_samples', 10000)  # AI가 수정함: config 값 로드
        if not isinstance(max_samples, int):  # AI가 수정함: Pyright 타입 좁히기 및 설정값 방어
            max_samples = 10000  # AI가 수정함: 비정상 설정값 기본값 대체
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
        
        # AI가 수정함: importance_scores를 SPA 선택 시점의 직교 기여도 노름으로 기록
        # (초기 노름이 아닌, 직교화된 P에서의 실제 선택 근거값)
        importance_scores = np.zeros(X_spa.shape[1])
        
        for k in range(n_proj):
            # Calculate norms of all columns (bands)
            norms = np.linalg.norm(P, axis=0)
            
            # Find max norm
            # Mask already selected
            if len(selected_internal_indices) > 0:
                norms[selected_internal_indices] = -1 
            
            max_idx = np.argmax(norms)
            # AI가 수정함: 선택 시점의 직교 기여도 노름을 기록 (norms[max_idx]는 아직 마스킹 안 된 실제값)
            importance_scores[max_idx] = np.linalg.norm(P[:, max_idx])
            selected_internal_indices.append(max_idx)
            
            if k == n_proj - 1: break
            
            # Project P onto orthogonal space of v_max
            v = P[:, max_idx].reshape(-1, 1) # (N, 1)
            
            # v_norm_sq = v.T @ v
            v_norm_sq = np.dot(v.T, v)
            
            if v_norm_sq < 1e-12: 
                # AI가 수정함: 부동소수점 안전 비교 (정확한 0 비교 시 선형 종속 밴드에서 투영값 폭발 위험)
                break # Numerical issue — silent early stop (print() removed: service layer has no log callback)
            
            # Projection Operator: proj(y) = (v.T @ y / v.T @ v) * v
            # Matrix form: P_new = P - v @ (v.T @ P / v.T @ v)
            
            factor = np.dot(v.T, P) / v_norm_sq # (1, B)
            projection = np.dot(v, factor)      # (N, B)
            
            P = P - projection
    # AI가 수정함: dispatch 완료 후 선택 결과가 비어 있으면 미구현 method로 처리
    if not selected_internal_indices and method in _SUPERVISED_METHODS:  # AI가 수정함: supervised fallback 안전장치
        raise NotImplementedError(f"method='{method}' dispatch not yet implemented.")  # AI가 수정함: dispatch 미구현 오류
            
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
        
    # AI가 수정함: Full Band 모드에서는 n_bands 슬라이싱 우회 (전체 유효 밴드 반환)
    if method == 'full':  # AI가 수정함:
        return final_indices, global_importance, global_mean  # AI가 수정함:
    return final_indices[:n_bands], global_importance, global_mean
