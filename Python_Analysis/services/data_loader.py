import os
import spectral.io.envi as envi
import numpy as np

def load_hsi_data(header_path: str) -> tuple:
    """
    Loads ENVI HSI data.
    Returns: (cube, wavelengths)
    """
    if not os.path.exists(header_path):
        # AI가 수정함: Strict Mode - 파일 없으면 에러 발생 (학습 데이터 오염 방지)
        raise FileNotFoundError(f"HSI Data file not found: {header_path}")
        
    try:
        img = envi.open(header_path)
        img_obj = img.load()
        data_cube = np.array(img_obj)
        
        metadata = img.metadata
        if 'wavelength' in metadata:
            wavelengths = [float(w) for w in metadata['wavelength']]
        else:
            wavelengths = list(range(data_cube.shape[2]))
            
        if 'wavelength' in metadata:
            wavelengths = [float(w) for w in metadata['wavelength']]
        else:
            wavelengths = list(range(data_cube.shape[2]))
            
        # Removed auto-scaling logic. Dealing with raw values faithfully.

        return data_cube, wavelengths
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise
