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

def apply_simple_derivative(data, gap=5):
    """
    Apply Simple Derivative (Gap Difference).
    Formula: D[i] = Spectrum[i] - Spectrum[i - gap]
    
    Args:
        data: (Samples, Bands)
        gap: separation between bands (default: 5)
        
    Returns:
        Processed data (same shape). 
        First 'gap' elements are padded (e.g. repeated or zeroed).
    """
    if gap < 1: return data
    
    # data shape: (N, Bands)
    # We want: data[:, gap:] - data[:, :-gap]
    
    # Example: Bands=10, Gap=2
    # Result[2] = B[2] - B[0]
    # Result[3] = B[3] - B[1]
    # ...
    # Result[0], Result[1] are undefined (Pad with 0 or edge)
    
    diff = data[:, gap:] - data[:, :-gap]
    
    # Pad the beginning to maintain shape
    # Pad width: ((0,0), (gap, 0)) -> (samples_margin, bands_margin)
    padded = np.pad(diff, ((0, 0), (gap, 0)), mode='edge')
    
    return padded
