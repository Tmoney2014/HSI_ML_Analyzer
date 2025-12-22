import os
import spectral.io.envi as envi
import numpy as np
from models.hsi_data import HSIData

def load_hsi_data(header_path: str) -> tuple:
    """
    Loads ENVI HSI data.
    Returns: (cube, wavelengths)
    """
    if not os.path.exists(header_path):
        print(f"File not found: {header_path}, returning random data.")
        return np.random.rand(100, 100, 224), [float(i) for i in range(224)]
        
    try:
        img = envi.open(header_path)
        img_obj = img.load()
        data_cube = np.array(img_obj)
        
        metadata = img.metadata
        if 'wavelength' in metadata:
            wavelengths = [float(w) for w in metadata['wavelength']]
        else:
            wavelengths = list(range(data_cube.shape[2]))
            
        # --- Auto-Scaling for 12-bit Data ---
        # If max value is small (<= 4095), it's likely 12-bit native data.
        # But existing rules expect 16-bit range (e.g. > 33000).
        # So we scale it up: 12-bit << 4 = 16-bit
        # Using a safe heuristic: if max < 4096 + epsilon (allow for some noise or saturation)
        current_max = np.max(data_cube)
        if current_max > 0 and current_max <= 4095:
            print(f"[Info] 12-bit data detected (Max: {current_max}). Scaling to 16-bit.")
            # Scale: x * (65535 / 4095) approx x * 16
            data_cube = data_cube * 16.0 
            # Or use bit shift if integer, but here we might have converted to float/numpy default
        # ------------------------------------

        return data_cube, wavelengths
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise
