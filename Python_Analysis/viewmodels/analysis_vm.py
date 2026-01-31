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
        self._cached_file_path = None
        self._cached_ref_cube = None # (H, W, B) after Ref convert (expensive)
        self._cached_mask_params = None # (threshold, mask_rules)
        self._cached_masked_mean = None # (B,) Mean Spectrum BEFORE Preprocessing
        self._cached_prep_chain = None # For change detection
        
        # Connect to main VM signals
        self.main_vm.mode_changed.connect(self.on_mode_changed)
        
    def on_mode_changed(self, mode):
        # Trigger re-processing when mode changes
        self._cached_ref_cube = None # Mode changed -> Ref/Abs/Raw changed -> Invalidate Cude
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
            self.params_changed.emit()
            self.model_updated.emit()
        
    def set_mask_rules(self, rules: str):
        if self.mask_rules != rules:
            self.mask_rules = rules
            self._cached_masked_mean = None # Mask changed -> Mean changed
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
                pass # Ignore bad input
                
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
            
            # If Ref Cache is missing (First load or Mode changed), we must compute it
            if file_changed or self._cached_ref_cube is None:
                # 1. Load Raw
                if file_path in self.main_vm.data_cache:
                    raw_cube, waves = self.main_vm.data_cache[file_path]
                else:
                    raw_cube, waves = load_hsi_data(file_path)
                    raw_cube = np.nan_to_num(raw_cube)
                    self.main_vm.data_cache[file_path] = (raw_cube, waves)
                
                # 2. Mode Conversion (Raw -> Ref, or Raw -> Abs)
                mode = self.processing_mode
                if mode == "Reflectance":
                    self._cached_ref_cube = self._convert_to_ref(raw_cube)
                elif mode == "Absorbance":
                    # Optimization: create_background_mask might need Ref or Abs? 
                    # Usually mask is done on Intensity (Raw or Ref). 
                    # But Absorbance is -log(Ref).
                    # Current logic: Ref -> Abs
                    ref_cube = self._convert_to_ref(raw_cube)
                    self._cached_ref_cube = processing.apply_absorbance(ref_cube)
                else: # Raw
                    self._cached_ref_cube = raw_cube
                
                # Update Cache State
                self._cached_file_path = file_path
                self._cached_masked_mean = None # New data -> New mean needed
            
            # If we are here, self._cached_ref_cube has the correct Cube (Raw/Ref/Abs)
            # Retrieve waves if not loaded above
            if waves is None:
                 if file_path in self.main_vm.data_cache:
                     _, waves = self.main_vm.data_cache[file_path]
            
            # -------------------------------------------------------------
            # LEVEL 2: Masking & Mean Calculation (Heavy Boolean Ops)
            # -------------------------------------------------------------
            
            # Check if we need to re-calculate Mean
            # Conditions: Mean cache empty OR Mask Params changed
            # current_mask_params = (self.threshold, self.mask_rules)
            
            # Note: We rely on _cached_masked_mean being cleared by setters.
            
            if self._cached_masked_mean is None:
                cube_to_mask = self._cached_ref_cube
                
                # Create mask (Always returns boolean mask)
                mask = processing.create_background_mask(cube_to_mask, self.threshold, self.mask_rules)
                valid_pixels = processing.apply_mask(cube_to_mask, mask)
                
                # print(f"[DEBUG] Re-Masking: Th={self.threshold}, Pixels={valid_pixels.shape[0]}")
                
                if valid_pixels.size == 0:
                    # print("[WARN] All pixels masked!")
                    return None, waves
                    
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
        # 1. Lazy Load White Ref
        if self.main_vm.cache_white is None and self.main_vm.white_ref:
            try:
                print(f"[DEBUG] Lazy Loading White Ref: {self.main_vm.white_ref}")
                w_data, _ = load_hsi_data(self.main_vm.white_ref)
                if w_data is None:
                    print("[DEBUG] White Ref Load Failed (None)")
                else:
                    print(f"[DEBUG] White Ref Data Shape: {w_data.shape}")
                    # Average to 1D pattern (mean of spatial pixels)
                    # Assumes w_data is (H, W, Bands) or (N, Bands)
                    # We need a 1D vector (Bands,)
                    w_vec = np.nanmean(w_data.reshape(-1, w_data.shape[-1]), axis=0)
                    self.main_vm.cache_white = w_vec
                    print(f"[DEBUG] Cache White Set. Vector Shape: {w_vec.shape}")
            except Exception as e:
                print(f"[ERROR] Error loading White Ref: {e}")
                
        # 2. Lazy Load Dark Ref
        if self.main_vm.cache_dark is None and self.main_vm.dark_ref:
            try:
                print(f"[DEBUG] Lazy Loading Dark Ref: {self.main_vm.dark_ref}")
                d_data, _ = load_hsi_data(self.main_vm.dark_ref)
                d_vec = np.nanmean(d_data.reshape(-1, d_data.shape[-1]), axis=0)
                self.main_vm.cache_dark = d_vec
                print(f"[DEBUG] Cache Dark Set.")
            except Exception as e:
                print(f"[ERROR] Error loading Dark Ref: {e}")

        # 3. Apply
        w_vec = self.main_vm.cache_white
        d_vec = self.main_vm.cache_dark
        
        if w_vec is None: 
            # print("[DEBUG] White Ref is None, returning Raw") # Too noisy for loop
            return raw_cube
        
        d = d_vec if d_vec is not None else np.zeros_like(w_vec)
        
        # Check shapes
        if raw_cube.shape[-1] != w_vec.shape[0]:
            print(f"[ERROR] Band Mismatch: Cube {raw_cube.shape} vs White {w_vec.shape}")
            return raw_cube

        numerator = raw_cube - d
        denominator = w_vec - d
        
        # Avoid div by zero
        denominator[denominator == 0] = 1e-6
        
        return np.clip(numerator / denominator, 0.0, 1.0)
