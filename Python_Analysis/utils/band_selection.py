import numpy as np
from sklearn.decomposition import PCA

def select_best_bands(data_cube, n_bands=5, method='pca'):
    """
    Selects the most informative bands from the data cube using PCA.
    It selects bands that have the highest correlation with the top Principal Components.
    
    Args:
        data_cube: 3D HSI data (Rows, Cols, Bands)
        n_bands: Number of bands to select
        method: 'pca' (currently only PCA is implemented)
        
    Returns:
        selected_band_indices: List of indices for the selected bands
    """
    print(f"   [Band Selection] Running {method.upper()} to select {n_bands} bands...")
    
    h, w, b = data_cube.shape
    
    # Flatten the data to (N_samples, N_features) -> (H*W, B)
    # We use a subset if the image is too large to speed up PCA
    n_samples = 10000
    flat_data = data_cube.reshape(-1, b)
    
    if flat_data.shape[0] > n_samples:
        indices = np.random.choice(flat_data.shape[0], n_samples, replace=False)
        flat_data_subset = flat_data[indices, :]
    else:
        flat_data_subset = flat_data
        
    # Apply PCA
    pca = PCA(n_components=n_bands)
    pca.fit(flat_data_subset)
    
    # Analyze components to find key bands
    # components_ shape is (n_components, n_features)
    # We take the argmax of the absolute value of components to find the band with max contribution per PC
    selected_band_indices = set()
    
    # Strategy: For each PC, find the band with the highest loading
    abs_components = np.abs(pca.components_)
    
    for i in range(min(n_bands, abs_components.shape[0])):
        # Find band with max coefficient in this component
        band_idx = np.argmax(abs_components[i])
        selected_band_indices.add(int(band_idx))
        
    # If we need more bands (due to duplicates), fill with next highest loadings
    if len(selected_band_indices) < n_bands:
        flat_loadings = np.sum(abs_components, axis=0)
        sorted_indices = np.argsort(flat_loadings)[::-1]
        for idx in sorted_indices:
            selected_band_indices.add(int(idx))
            if len(selected_band_indices) >= n_bands:
                break
                
    result_list = sorted(list(selected_band_indices))[:n_bands]
    print(f"   [Band Selection] Selected Indices: {result_list}")
    
    return result_list
