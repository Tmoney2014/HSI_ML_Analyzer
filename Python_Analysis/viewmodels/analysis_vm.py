from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from viewmodels.main_vm import MainViewModel
from models import processing
from services.data_loader import load_hsi_data

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
        
        # Connect to main VM signals
        self.main_vm.mode_changed.connect(self.on_mode_changed)
        
    def on_mode_changed(self, mode):
        # Trigger re-processing when mode changes
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
        self.params_changed.emit()
        self.model_updated.emit()

    def set_full_state(self, state):
        """Set complete state from UI (includes disabled items)."""
        self._full_state = state
        self.params_changed.emit()
        self.model_updated.emit()
        
    def get_full_state(self):
        return self._full_state

    # Deprecated/Legacy method support (mapped to setter)
    def set_preprocessing_chain(self, chain):
        self.prep_chain = chain

    def set_threshold(self, val: float):
        self.threshold = val
        self.params_changed.emit()
        self.model_updated.emit()
        
    def set_mask_rules(self, rules: str):
        self.mask_rules = rules
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
        self.params_changed.emit()
        self.model_updated.emit()
        
    def remove_step(self, index):
        if 0 <= index < len(self.prep_chain):
            del self.prep_chain[index]
            self.params_changed.emit()
            self.model_updated.emit()
            
    def move_step(self, index, direction):
        if direction == -1 and index > 0:
            self.prep_chain[index], self.prep_chain[index-1] = self.prep_chain[index-1], self.prep_chain[index]
            self.params_changed.emit()
            self.model_updated.emit()
        elif direction == 1 and index < len(self.prep_chain) - 1:
            self.prep_chain[index], self.prep_chain[index+1] = self.prep_chain[index+1], self.prep_chain[index]
            self.params_changed.emit()
            self.model_updated.emit()

    def update_params(self, index, new_params):
        if 0 <= index < len(self.prep_chain):
            self.prep_chain[index]['params'] = new_params
            self.params_changed.emit()
            self.model_updated.emit()
        
    def get_processed_spectrum(self, file_path):
        try:
            # 1. Load Data (Cache Check)
            if file_path in self.main_vm.data_cache:
                cube, waves = self.main_vm.data_cache[file_path]
            else:
                cube, waves = load_hsi_data(file_path)
                cube = np.nan_to_num(cube)
                # Cache it
                self.main_vm.data_cache[file_path] = (cube, waves)
            
            # 2. Mode Conversion (Reflectance / Absorbance)
            mode = self.processing_mode
            if mode == "Reflectance":
                cube = self._convert_to_ref(cube)
            elif mode == "Absorbance":
                cube = self._convert_to_ref(cube)
                cube = processing.apply_absorbance(cube)
                
            # 3. Masking
            mask = processing.create_background_mask(cube, self.threshold, self.mask_rules)
            valid_pixels = processing.apply_mask(cube, mask)
            
            print(f"[DEBUG] Mode: {mode}, Threshold: {self.threshold}, MaskRules: {self.mask_rules}")
            print(f"[DEBUG] Total Pixels: {cube.shape[0]*cube.shape[1]}, Valid Pixels: {valid_pixels.shape[0]}")

            if valid_pixels.size == 0:
                print("[WARN] All pixels masked! Check Threshold.")
                return None, waves
                
            # 4. Mean Spectrum
            mean_spec = np.mean(valid_pixels, axis=0)
            
            # 5. Apply Chain
            processed = mean_spec[np.newaxis, :] # Make 2D (1, Bands) for functions
            
            for step in self.prep_chain:
                name = step.get('name')
                p = step.get('params', {})
                
                if name == "SG":
                    processed = processing.apply_savgol(processed, 
                                                        window_size=p.get('win', 5), 
                                                        poly_order=p.get('poly', 2),
                                                        deriv=p.get('deriv', 0))
                elif name == "SimpleDeriv":
                    processed = processing.apply_simple_derivative(processed, gap=p.get('gap', 5), order=p.get('order', 1), apply_ratio=p.get('ratio', False), ndi_threshold=p.get('ndi_threshold', 1e-4))
                elif name == "SNV":
                    processed = processing.apply_snv(processed)
                elif name == "L2":
                    processed = processing.apply_l2_norm(processed)
                elif name == "3PointDepth":
                    processed = processing.apply_rolling_3point_depth(processed, gap=p.get('gap', 5))
                elif name == "MinMax":
                    processed = processing.apply_minmax_norm(processed)
                elif name == "MinSub":
                    processed = processing.apply_min_subtraction(processed)
                elif name == "Center":
                    processed = processing.apply_mean_centering(processed)
                
                # Absorbance step removed from list logic
                
                processed = np.nan_to_num(processed)
                
            return processed.flatten(), waves
            
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
