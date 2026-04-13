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
                # AI가 수정함: apply_absorbance() 단일 경로 통일 — P-1
                # 이전: np.where(<=0, 1e-6) + -np.log10() 인라인 구현
                # 문제: processing.apply_absorbance()와 코드 경로 이원화 → 한쪽만 수정 시 parity 파괴
                flat_data = processing.apply_absorbance(flat_data)  # epsilon=1e-6 기본값
            
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
    def parse_raw_band_indices(exclude_bands_str: str) -> list:
        """
        Parse raw-band exclude input like "1-5, 92" into sorted 0-based raw indices.

        UI contract: user input is always RAW sensor band numbering (1-based).
        """
        if not exclude_bands_str:
            return []

        indices = set()
        for part in exclude_bands_str.split(','):
            token = part.strip()
            if not token:
                continue

            if '-' in token:
                start_str, end_str = token.split('-', 1)
                start = int(start_str)
                end = int(end_str)
                if start <= 0 or end <= 0 or end < start:
                    raise ValueError(
                        f"Invalid raw band range: '{token}'. Use 1-based positive ranges like '1-10'."
                    )
                for idx in range(start, end + 1):
                    indices.add(idx - 1)
            else:
                idx = int(token)
                if idx <= 0:
                    raise ValueError(
                        f"Invalid raw band index: '{token}'. Use 1-based positive integers."
                    )
                indices.add(idx - 1)

        return sorted(indices)

    @staticmethod
    def _build_processed_raw_dependencies(raw_band_count: int, prep_chain: list) -> list:
        """
        Build processed-feature -> raw-band dependency sets for a preprocessing chain.

        Returns:
            dependencies: list[set[int]] where dependencies[i] is the RAW band set
            required to produce processed feature i after applying prep_chain in order.
        """
        if raw_band_count <= 0:
            return []

        dependencies = [{i} for i in range(raw_band_count)]

        for step in prep_chain or []:
            name = step.get('name')
            params = step.get('params', {})

            if name == "SG":
                window_size = int(params.get('win', 5))
                if window_size % 2 == 0:
                    window_size += 1
                radius = max(window_size // 2, 0)
                last = len(dependencies) - 1
                expanded = []
                for i in range(len(dependencies)):
                    dep = set()
                    left = max(0, i - radius)
                    right = min(last, i + radius)
                    for j in range(left, right + 1):
                        dep.update(dependencies[j])
                    expanded.append(dep)
                dependencies = expanded

            elif name == "SimpleDeriv":
                gap = int(params.get('gap', 5))
                order = int(params.get('order', 1))
                if gap < 1 or order < 1:
                    continue
                for _ in range(order):
                    if len(dependencies) <= gap:
                        raise ValueError(
                            f"Strict Mode Error: Not enough bands to build derivative dependencies. "
                            f"Has {len(dependencies)}, Need > {gap}."
                        )
                    dependencies = [
                        dependencies[i] | dependencies[i + gap]
                        for i in range(len(dependencies) - gap)
                    ]

            elif name in {"SNV", "L2", "MinSub", "MinMax", "Absorbance", None}:
                # Positional mapping preserved for these transforms.
                continue

            else:
                # Unknown/legacy transforms are treated as position-preserving by default.
                continue

        return dependencies

    @staticmethod
    def map_raw_excludes_to_processed_indices(raw_exclude_indices, raw_band_count: int, prep_chain: list) -> list:
        """
        Map RAW-band exclusions to processed feature indices.

        Policy:
        - User exclusions are always RAW sensor band numbers.
        - A processed feature is excluded if its positional RAW dependency intersects
          any excluded RAW band.
        - Global spectrum normalizers (SNV/L2/MinMax/MinSub) preserve positional mapping
          for exclusion purposes; they do not remap feature indices.
        """
        if raw_band_count <= 0 or not raw_exclude_indices:
            return []

        raw_exclude_set = {int(i) for i in raw_exclude_indices if 0 <= int(i) < raw_band_count}
        if not raw_exclude_set:
            return []

        dependencies = ProcessingService._build_processed_raw_dependencies(raw_band_count, prep_chain)

        return [i for i, dep in enumerate(dependencies) if dep & raw_exclude_set]

    @staticmethod
    def map_processed_indices_to_raw_dependencies(processed_indices, raw_band_count: int, prep_chain: list) -> list:
        """
        Map processed feature indices back to the sorted unique RAW sensor band indices
        required to reproduce them at runtime.
        """
        if raw_band_count <= 0 or not processed_indices:
            return []

        dependencies = ProcessingService._build_processed_raw_dependencies(raw_band_count, prep_chain)
        required_raw = set()

        for idx in processed_indices:
            p_idx = int(idx)
            if 0 <= p_idx < len(dependencies):
                required_raw.update(dependencies[p_idx])
            else:
                raise ValueError(
                    f"Processed feature index out of range for dependency mapping: {p_idx} "
                    f"(processed feature count={len(dependencies)})"
                )

        return sorted(required_raw)

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
                    order=ProcessingService._req(p, 'order', 'SimpleDeriv')
                )
            elif name == "SNV": 
                flat_data = processing.apply_snv(flat_data)
            elif name == "L2": 
                flat_data = processing.apply_l2_norm(flat_data)
            elif name == "MinSub": 
                flat_data = processing.apply_min_subtraction(flat_data)
            elif name == "MinMax": 
                flat_data = processing.apply_minmax_norm(flat_data)
            elif name == "Absorbance":
                # AI가 추가함: Absorbance step 이중 변환 방지 — P-4
                # Absorbance 변환은 get_base_data()에서 mode='Absorbance' 시 자동 처리됨.
                # prep_chain에 명시적으로 포함하면 이중 변환(−log10(−log10(R))) 발생.
                raise ValueError(
                    "Strict Mode Error: 'Absorbance' step은 prep_chain에 포함할 수 없습니다. "
                    "Absorbance 변환은 mode='Absorbance' 설정으로 get_base_data()에서 자동 처리됩니다."
                )
            else:
                # AI가 추가함: 알 수 없는 step name 경고 (silent skip 방지)
                print(f"Warning: [ProcessingService] Unknown prep step '{name}' — skipped. Check prep_chain configuration.")
        
        return flat_data
