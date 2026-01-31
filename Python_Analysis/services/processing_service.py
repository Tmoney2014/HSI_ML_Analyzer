import numpy as np
from models import processing

class ProcessingService:
    """
    Shared service for HSI data processing pipeline.
    Ensures consistency between Analysis View and Training View.
    """
    
    @staticmethod
    def process_cube(cube: np.ndarray, 
                     mode: str, 
                     threshold: float, 
                     mask_rules: str, 
                     prep_chain: list,
                     white_ref: np.ndarray = None,
                     dark_ref: np.ndarray = None) -> tuple:
        """
        Full pipeline: Raw Cube -> Ref/Abs -> Mask -> Preprocessing -> Spectrum
        
        Args:
            cube: Raw HSI Cube (H, W, B)
            mode: "Raw", "Reflectance", "Absorbance"
            threshold: Mask threshold
            mask_rules: Advanced mask rules string
            prep_chain: List of prep steps [{'name': 'SG', 'params': {...}}, ...]
            apply_absorbance_in_loop: (Legacy compat) If True, applies Absorbance AFTER Ref conversion 
                                      explicitly (TrainingVM logic). AnalysisVM does it via prep chain usually.
                                      Actually, AnalysisVM logic: Ref -> Abs (if mode=Abs) -> Mask -> Prep.
                                      
        Returns:
            (processed_data, valid_mask)
            processed_data: (N_pixels, B) 2D array of valid pixels
            valid_mask: (H, W) boolean mask
        """
        # Call Helper to get Base Data (Reflectance + Masked)
        flat_data, mask = ProcessingService.get_base_data(
             cube, mode, threshold, mask_rules, white_ref, dark_ref
        )
        
        # 5. Preprocessing Chain
        flat_data = ProcessingService.apply_preprocessing_chain(flat_data, prep_chain)
        
        flat_data = np.nan_to_num(flat_data)
        
        return flat_data, mask

    @staticmethod
    def get_base_data(cube, mode, threshold, mask_rules, white_ref=None, dark_ref=None):
        """
        Generates Base Data (Masked valid pixels, Ref/Abs converted, but NO Preprocessing).
        """
        # 1. Input Sanitization
        cube = np.nan_to_num(cube)
        
        # 2. Convert to Reflectance / Absorbance
        data_cube = cube
        
        if mode in ["Reflectance", "Absorbance"]:
            if white_ref is not None and dark_ref is not None:
                denom = np.subtract(white_ref, dark_ref, dtype=np.float32)
                denom[denom == 0] = 1e-6 
                num = np.subtract(cube, dark_ref, dtype=np.float32)
                data_cube = np.divide(num, denom, dtype=np.float32)
            elif white_ref is not None:
                 denom = white_ref.astype(np.float32)
                 denom[denom == 0] = 1e-6
                 data_cube = cube.astype(np.float32) / denom
        
        # Standardize Absorbance Logic
        if mode == "Absorbance":
             # Apply log(1/R) -> -log(R)
             data_cube = np.where(data_cube <= 0, 1e-6, data_cube)
             data_cube = -np.log10(data_cube)
            
        # 3. Masking
        mask = processing.create_background_mask(data_cube, threshold, mask_rules)
        if mask.dtype != bool: 
            mask = mask.astype(bool)
            
        # 4. Apply Mask
        # Returns (N, B)
        flat_data = processing.apply_mask(data_cube, mask)
        
        return flat_data, mask

    @staticmethod
    def apply_preprocessing_chain(flat_data, prep_chain):
        """
        Apply a list of preprocessing steps to flattened data (N, Bands).
        """
        for step in prep_chain:
            name = step.get('name')
            p = step.get('params', {})
            
            if name == "SG": 
                flat_data = processing.apply_savgol(flat_data, p.get('win'), p.get('poly'), p.get('deriv', 0))
            elif name == "SimpleDeriv": 
                flat_data = processing.apply_simple_derivative(
                    flat_data, 
                    gap=p.get('gap', 5), 
                    order=p.get('order', 1), 
                    apply_ratio=p.get('ratio', False), 
                    ndi_threshold=p.get('ndi_threshold', 1e-4)
                )
            elif name == "SNV": 
                flat_data = processing.apply_snv(flat_data)
            elif name == "3PointDepth": 
                flat_data = processing.apply_rolling_3point_depth(flat_data, gap=p.get('gap', 5))
            elif name == "L2": 
                flat_data = processing.apply_l2_norm(flat_data)
            elif name == "MinSub": 
                flat_data = processing.apply_min_subtraction(flat_data)
            elif name == "MinMax": 
                flat_data = processing.apply_minmax_norm(flat_data)
            elif name == "Center": 
                flat_data = processing.apply_mean_centering(flat_data)
        
        return flat_data
