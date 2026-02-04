from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from viewmodels.main_vm import MainViewModel
from models import processing
from services.data_loader import load_hsi_data
from services.processing_service import ProcessingService

class AnalysisViewModel(QObject):
    visualization_updated = pyqtSignal(object, object) # (spec_data, waves)
    error_occurred = pyqtSignal(str)
    # Signal for UI updates
    model_updated = pyqtSignal() # When processing chain or params change significantly affecting visualization
    params_changed = pyqtSignal() # When any parameter changes (for Auto-Save)
    # AI가 수정함: Base Data에 영향을 주는 변경만 별도 시그널로 분리 (Gap 등 전처리 변경은 제외)
    base_data_invalidated = pyqtSignal() # When threshold, mask_rules change (actual base data impact)

    def __init__(self, main_vm: MainViewModel):
        super().__init__()
        self.main_vm = main_vm
        
        # Analysis parameters
        self.threshold = 0.0 # Will be updated by UI
        self.mask_rules = None # e.g. "Band 10 > 500" or just "Mean"
        self.exclude_bands_str = "" # User input for exclusion
        # self.use_ref delegated to MainVM
        
        # Preprocessing Chain
        # Full State: List of dicts: {"name": "SG", "params": {...}, "enabled": bool}
        self._full_state = []
        
        # --- Caching State ---
        # AI가 수정함: Raw Cube를 별도로 캐시하여 마스킹 기준 유지
        self._cached_file_path = None
        self._cached_raw_cube = None   # (H, W, B) Raw Cube (마스킹용)
        self._cached_ref_cube = None   # (H, W, B) 시각화용 (Ref/Abs 변환 후)
        self._cached_mask_params = None # (threshold, mask_rules)
        self._cached_masked_mean = None # (B,) Mean Spectrum BEFORE Preprocessing
        self._cached_prep_chain = None # For change detection
        
        # Connect to main VM signals
        self.main_vm.mode_changed.connect(self.on_mode_changed)
        
    def on_mode_changed(self, mode):
        # Trigger re-processing when mode changes
        # AI가 수정함: Raw Cube는 유지, 변환된 결과만 무효화
        self._cached_ref_cube = None  # Mode changed -> 변환 결과만 무효화
        self._cached_masked_mean = None 
        self.params_changed.emit()
        self.model_updated.emit()
        
    @property
    def processing_mode(self):
        return self.main_vm.processing_mode

    @property
    def use_ref(self):
        # Compatibility property
        return self.processing_mode in ["Reflectance", "Absorbance"]
        
    @property
    def prep_chain(self):
        """Returns only enabled steps for processing engine."""
        return [{"name": s["name"], "params": s["params"]} for s in self._full_state if s.get("enabled", False)]

    # ... (skipping setter) ...

    # ...

    @prep_chain.setter
    def prep_chain(self, chain):
        """Legacy Setter: Reconstructs full state from active chain list."""
        self._full_state = []
        for step in chain:
            new_step = step.copy()
            new_step["enabled"] = True
            self._full_state.append(new_step)
        self._invalidate_processing()
        self.params_changed.emit()
        self.model_updated.emit()

    def set_full_state(self, state):
        """Set complete state from UI (includes disabled items)."""
        self._full_state = state
        self._invalidate_processing()
        self.params_changed.emit()
        self.model_updated.emit()
        
    def get_full_state(self):
        return self._full_state
        
    def _invalidate_processing(self):
        """Call when only Preprocessing chain changes (Mask/Ref unchanged)"""
        # We don't need to clear _cached_masked_mean
        pass

    # Deprecated/Legacy method support (mapped to setter)
    def set_preprocessing_chain(self, chain):
        self.prep_chain = chain

    def set_threshold(self, val: float):
        if self.threshold != val:
            self.threshold = val
            self._cached_masked_mean = None # Mask changed -> Mean changed
            self.base_data_invalidated.emit()  # AI가 수정함: Base Data 무효화 시그널
            self.params_changed.emit()
            self.model_updated.emit()
        
    def set_mask_rules(self, rules: str):
        if self.mask_rules != rules:
            self.mask_rules = rules
            self._cached_masked_mean = None # Mask changed -> Mean changed
            self.base_data_invalidated.emit()  # AI가 수정함: Base Data 무효화 시그널
            self.params_changed.emit()
            self.model_updated.emit()
        
    def set_exclude_bands(self, val: str):
        self.exclude_bands_str = val
        self.params_changed.emit() # Auto-save trigger if needed
        
    def parse_exclude_bands(self):
        """
        Parses "1-5, 92" string into list of 0-based indices.
        """
        if not self.exclude_bands_str: return []
        
        indices = set()
        parts = self.exclude_bands_str.split(',')
        for p in parts:
            p = p.strip()
            if not p: continue
            
            try:
                if '-' in p:
                    # Range: "1-5"
                    start, end = p.split('-')
                    s = int(start)
                    e = int(end)
                    # User 1-based -> 0-based
                    # Range inclusive for user
                    for i in range(s, e + 1):
                        indices.add(i - 1)
                else:
                    # Single: "92"
                    indices.add(int(p) - 1)
            except ValueError:
                # AI가 수정함: Strict Mode - 입력 에러 알림
                msg = f"Invalid format in exclude bands: '{p}'. Use numbers or ranges (e.g. 1-10)."
                self.error_occurred.emit(msg)
                return [] # 중단
                
        return list(indices)
        
    def add_step(self, step_name, params=None):
        step = {"name": step_name, "params": params or {}}
        self.prep_chain.append(step)
        self._invalidate_processing()
        self.params_changed.emit()
        self.model_updated.emit()
        
    def remove_step(self, index):
        if 0 <= index < len(self.prep_chain):
            del self.prep_chain[index]
            self._invalidate_processing()
            self.params_changed.emit()
            self.model_updated.emit()
            
    def move_step(self, index, direction):
        if direction == -1 and index > 0:
            self.prep_chain[index], self.prep_chain[index-1] = self.prep_chain[index-1], self.prep_chain[index]
            self._invalidate_processing()
            self.params_changed.emit()
            self.model_updated.emit()
        elif direction == 1 and index < len(self.prep_chain) - 1:
            self.prep_chain[index], self.prep_chain[index+1] = self.prep_chain[index+1], self.prep_chain[index]
            self._invalidate_processing()
            self.params_changed.emit()
            self.model_updated.emit()

    def update_params(self, index, new_params):
        if 0 <= index < len(self.prep_chain):
            self.prep_chain[index]['params'] = new_params
            self._invalidate_processing()
            self.params_changed.emit()
            self.model_updated.emit()
        
    def get_processed_spectrum(self, file_path):
        try:
            waves = None
            
            # -------------------------------------------------------------
            # LEVEL 1: File Check & Mode Check (Reflectance Transformation)
            # -------------------------------------------------------------
            
            # If file changed, we must reload (and invalidates everything)
            file_changed = (file_path != self._cached_file_path)
            
            # AI가 수정함: Raw Cube와 변환된 Cube 분리 캐싱
            if file_changed or self._cached_raw_cube is None:
                # 1. Load Raw
                if file_path in self.main_vm.data_cache:
                    raw_cube, waves = self.main_vm.data_cache[file_path]
                else:
                    raw_cube, waves = load_hsi_data(file_path)
                    raw_cube = np.nan_to_num(raw_cube)
                    self.main_vm.data_cache[file_path] = (raw_cube, waves)
                
                # AI가 수정함: Raw Cube 별도 캐시 (마스킹용)
                self._cached_raw_cube = raw_cube
                self._cached_file_path = file_path
                self._cached_ref_cube = None  # 새 파일 -> 변환 결과 무효화
                self._cached_masked_mean = None  # New data -> New mean needed
            
            # 시각화용 변환된 Cube 캐시 (Mode에 따라)
            if self._cached_ref_cube is None:
                mode = self.processing_mode
                if mode == "Reflectance":
                    self._cached_ref_cube = self._convert_to_ref(self._cached_raw_cube)
                elif mode == "Absorbance":
                    ref_cube = self._convert_to_ref(self._cached_raw_cube)
                    self._cached_ref_cube = processing.apply_absorbance(ref_cube)
                else:  # Raw
                    self._cached_ref_cube = self._cached_raw_cube
            
            # Retrieve waves if not loaded above
            if waves is None:
                 if file_path in self.main_vm.data_cache:
                     _, waves = self.main_vm.data_cache[file_path]
            
            # -------------------------------------------------------------
            # LEVEL 2: Masking & Mean Calculation (Raw 기준 마스킹)
            # -------------------------------------------------------------
            # AI가 수정함: 설계 원칙 - 마스킹은 Raw 값 기준으로 수행
            
            if self._cached_masked_mean is None:
                # 1. Masking on RAW Cube (설계 원칙: MaskRules는 Raw DN 값 기준)
                mask = processing.create_background_mask(
                    self._cached_raw_cube, self.threshold, self.mask_rules
                )
                
                # 2. Apply Mask to get valid pixels from RAW
                valid_raw = processing.apply_mask(self._cached_raw_cube, mask)
                
                if valid_raw.size == 0:
                    return None, waves
                
                # 3. Convert valid pixels only (Raw -> Ref/Abs)
                mode = self.processing_mode
                if mode in ["Reflectance", "Absorbance"]:
                    valid_pixels = ProcessingService.convert_to_ref_flat(
                        valid_raw, 
                        self.main_vm.cache_white, 
                        self.main_vm.cache_dark
                    )
                    if mode == "Absorbance":
                        valid_pixels = processing.apply_absorbance(valid_pixels)
                else:
                    valid_pixels = valid_raw.astype(np.float32)
                
                # Compute Mean
                self._cached_masked_mean = np.mean(valid_pixels, axis=0)
                
            # -------------------------------------------------------------
            # LEVEL 3: Preprocessing Chain (Fast 1D Ops)
            # -------------------------------------------------------------
            
            # Use ProcessingService for consistency
            mean_spec = self._cached_masked_mean
            
            # Prepare for service: (1, 1, B)
            # We treat this mean spectrum as a "Raw Cube" for the service 
            # because we only want to apply the Prep Chain (Ref/Masking already done)
            dummy_cube = mean_spec.reshape(1, 1, -1)
            
            processed_cube, _ = ProcessingService.process_cube(
                dummy_cube, 
                mode="Raw",          # Don't re-convert
                threshold=-999.0,    # Don't re-mask
                mask_rules=None,
                prep_chain=self.prep_chain
            )
            
            # processed_cube is (1, B) (n_pixels=1 flattened)
            processed = processed_cube.flatten()
            
            return processed, waves
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            return None, None

    def _convert_to_ref(self, raw_cube):
        """
        /// AI가 수정함: 중복 로직 제거, 중앙화된 메서드로 위임
        Convert Raw Cube to Reflectance using centralized methods.
        """
        # 1. Ensure refs are loaded (centralized in MainViewModel)
        self.main_vm.ensure_refs_loaded()
        
        # 2. Convert using ProcessingService
        return ProcessingService.convert_to_ref(
            raw_cube, 
            self.main_vm.cache_white, 
            self.main_vm.cache_dark
        )

