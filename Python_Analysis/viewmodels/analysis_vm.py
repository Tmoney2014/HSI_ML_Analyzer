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
        # List of dicts: {"name": "SG", "params": {...}}
        self.prep_chain = []
        
        # Connect to main VM signals if needed
        # self.main_vm.files_changed.connect(self.on_data_changed)
        
    @property
    def use_ref(self):
        return self.main_vm.use_ref
        
    def set_preprocessing_chain(self, chain):
        self.prep_chain = chain
        self.params_changed.emit()
        self.model_updated.emit()

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
            
            # 2. Reflectance Conversion?
            if self.use_ref:
                cube = self._convert_to_ref(cube)
                
            # 3. Masking
            mask = processing.create_background_mask(cube, self.threshold, self.mask_rules)
            valid_pixels = processing.apply_mask(cube, mask)
            
            print(f"[Debug] File: {file_path}")
            print(f"[Debug] Threshold: {self.threshold}, Rules: {self.mask_rules}")
            print(f"[Debug] Cube Shape: {cube.shape}, Valid Pixels: {valid_pixels.shape[0]}")

            if valid_pixels.size == 0:
                print("[Debug] No valid pixels found!")
                return None, waves
                
            # 4. Mean Spectrum
            mean_spec = np.mean(valid_pixels, axis=0)
            
            # 5. Apply Chain
            # Expect chain items like: {'name': 'SG', 'params': {...}}
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
                
                processed = np.nan_to_num(processed)
                
            return processed.flatten(), waves
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            return None, None

    def _convert_to_ref(self, raw_cube):
        # Use Cached Vectors from MainVM
        w_vec = self.main_vm.cache_white
        d_vec = self.main_vm.cache_dark
        
        if w_vec is None: return raw_cube
        
        d = d_vec if d_vec is not None else 0
        
        numerator = raw_cube - d
        denominator = w_vec - d
        
        # Avoid div by zero
        denominator[denominator == 0] = 1e-6
        
        return np.clip(numerator / denominator, 0.0, 1.0)
