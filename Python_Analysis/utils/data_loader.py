import os
import spectral.io.envi as envi
import numpy as np

def load_hsi_data(header_path):
    """
    Loads HSI data from an ENVI header file.
    
    Args:
        header_path (str): Path to the .hdr file
        
    Returns:
        tuple: (data_cube, wavelengths)
            - data_cube: numpy array of shape (Rows, Cols, Bands)
            - wavelengths: list of wavelength values
    """
    if not os.path.exists(header_path):
        print(f"   [Data Loader] File not found: {header_path}")
        print("   [Data Loader] Creating synthetic data for demonstration...")
        # Create synthetic data: 100x100 pixels, 224 bands
        return np.random.rand(100, 100, 224), [float(i) for i in range(224)]
        
    print(f"   [Data Loader] Loading header: {header_path}")
    
    try:
        # Load the spectral image
        img = envi.open(header_path)
        
        # Load the whole data into memory
        data_cube = img.load()
        
        # Get wavelengths from metadata
        metadata = img.metadata
        if 'wavelength' in metadata:
            wavelengths = [float(w) for w in metadata['wavelength']]
        else:
            wavelengths = list(range(data_cube.shape[2]))
            
        print(f"   [Data Loader] Loaded Cube Shape: {data_cube.shape}")
        
        return data_cube, wavelengths
        
    except Exception as e:
        print(f"   [Error] Failed to load HSI data: {e}")
        raise
