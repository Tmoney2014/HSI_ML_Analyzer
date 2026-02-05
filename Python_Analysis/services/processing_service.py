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
    def convert_to_ref(cube, white_ref, dark_ref=None):
        """
        /// <ai>AI가 작성함</ai>
        Convert Raw Cube to Reflectance.
        Formula: R = (Sample - Dark) / (White - Dark)
        
        Args:
            cube: Raw HSI Cube (H, W, B) or flattened (N, B)
            white_ref: White reference vector (Bands,)
            dark_ref: Dark reference vector (Bands,), optional
        
        Returns:
            ref_cube: Reflectance values clipped to [0, 1]
        """
        if white_ref is None:
            return cube
        
        d = dark_ref if dark_ref is not None else np.zeros_like(white_ref)
        denom = (white_ref - d).astype(np.float32)
        denom[denom == 0] = 1e-6
        
        return np.clip((cube.astype(np.float32) - d) / denom, 0.0, 1.0)

    @staticmethod
    def convert_to_ref_flat(flat_data, white_ref, dark_ref=None):
        """
        /// <ai>AI가 작성함</ai>
        Convert flattened Raw data (N, B) to Reflectance.
        For use after masking when data is already flattened.
        
        Args:
            flat_data: Flattened raw data (N_pixels, Bands)
            white_ref: White reference vector (Bands,)
            dark_ref: Dark reference vector (Bands,), optional
        
        Returns:
            ref_data: Reflectance values clipped to [0, 1]
        """
        if white_ref is None:
            return flat_data.astype(np.float32)
        
        d = dark_ref if dark_ref is not None else np.zeros_like(white_ref)
        denom = (white_ref - d).astype(np.float32)
        denom[denom == 0] = 1e-6
        
        return np.clip((flat_data.astype(np.float32) - d) / denom, 0.0, 1.0)

    @staticmethod
    def get_base_data(cube, mode, threshold, mask_rules, white_ref=None, dark_ref=None):
        """
        /// <ai>AI가 수정함: 마스킹 순서를 Raw 단계로 변경</ai>
        Generates Base Data (Masked valid pixels, Ref/Abs converted, but NO Preprocessing).
        
        설계 원칙: 마스킹은 반드시 Raw 데이터 단계에서 수행.
        - MaskRules는 항상 Raw DN 값 기준 (예: b80 > 35000)
        - 성능: 배경 픽셀에 불필요한 Log 연산 회피
        - 일관성: Python 학습 ↔ C# 추론 동일 기준
        """
        try:
            # 1. Input Sanitization
            cube = np.nan_to_num(cube)
            
            # 2. Masking FIRST on Raw Cube
            mask = processing.create_background_mask(cube, threshold, mask_rules)
            
            if mask.dtype != bool: 
                mask = mask.astype(bool)
            
            # 3. Apply Mask to Raw Cube
            flat_raw = processing.apply_mask(cube, mask) 
            
            # 4. Convert valid pixels only to Ref/Abs
            if mode in ["Reflectance", "Absorbance"]:
                flat_data = ProcessingService.convert_to_ref_flat(flat_raw, white_ref, dark_ref)
            else:
                flat_data = flat_raw.astype(np.float32)
            
            # 5. Absorbance Transform
            if mode == "Absorbance":
                flat_data = np.where(flat_data <= 0, 1e-6, flat_data)
                flat_data = -np.log10(flat_data)
            
            return flat_data, mask
            
        except Exception as e:
            # print(f"DB [Error] get_base_data failed: {e}")
            raise e


    @staticmethod
    def _req(params: dict, key: str, step_name: str):
        """
        /// <ai>AI가 작성함</ai>
        Strict Param Getter.
        If key is missing, raises ValueError instead of using default.
        """
        if key not in params:
            raise ValueError(f"Strict Mode Error: Missing required param '{key}' in step '{step_name}'")
        return params[key]

    @staticmethod
    def apply_preprocessing_chain(flat_data, prep_chain):
        """
        Apply a list of preprocessing steps to flattened data (N, Bands).
        """
        for step in prep_chain:
            name = step.get('name')
            p = step.get('params', {})
            
            # AI가 수정함: Strict Mode 적용 (모든 파라미터 검증)
            if name == "SG": 
                flat_data = processing.apply_savgol(
                    flat_data, 
                    window_size=ProcessingService._req(p, 'win', 'SG'), 
                    poly_order=ProcessingService._req(p, 'poly', 'SG'), 
                    deriv=ProcessingService._req(p, 'deriv', 'SG')
                )
            elif name == "SimpleDeriv": 
                flat_data = processing.apply_simple_derivative(
                    flat_data, 
                    gap=ProcessingService._req(p, 'gap', 'SimpleDeriv'), 
                    order=ProcessingService._req(p, 'order', 'SimpleDeriv'), 
                    apply_ratio=p.get('ratio', False), # ratio는 boolean flag라 optional 가능 (하지만 UI에서 반드시 줌)
                    ndi_threshold=p.get('ndi_threshold', 1e-4) # Optional
                )
            elif name == "SNV": 
                flat_data = processing.apply_snv(flat_data)
            elif name == "3PointDepth": 
                flat_data = processing.apply_rolling_3point_depth(
                    flat_data, 
                    gap=ProcessingService._req(p, 'gap', '3PointDepth')
                )
            elif name == "L2": 
                flat_data = processing.apply_l2_norm(flat_data)
            elif name == "MinSub": 
                flat_data = processing.apply_min_subtraction(flat_data)
            elif name == "MinMax": 
                flat_data = processing.apply_minmax_norm(flat_data)
            elif name == "Center": 
                flat_data = processing.apply_mean_centering(flat_data)
        
        return flat_data
