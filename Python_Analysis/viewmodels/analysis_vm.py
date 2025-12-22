from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from viewmodels.main_vm import MainViewModel
from models import processing
from services.data_loader import load_hsi_data

class AnalysisViewModel(QObject):
    visualization_updated = pyqtSignal(object, object) # (spec_data, waves)
    error_occurred = pyqtSignal(str)

    def __init__(self, main_vm: MainViewModel):
        super().__init__()
        self.main_vm = main_vm
        
        # Params
        self.threshold = 0.1
        self.mask_rules = None
        # self.use_ref delegated to MainVM
        
        # Preprocessing Config
        self.prep_chain = [] # List of config dicts
        
    @property
    def use_ref(self):
        return self.main_vm.use_ref
        
    def set_preprocessing_chain(self, chain):
        self.prep_chain = chain
        
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
                    processed = processing.apply_simple_derivative(processed, gap=p.get('gap', 5))
                elif name == "SNV":
                    processed = processing.apply_snv(processed)
                elif name == "L2":
                    processed = processing.apply_l2_norm(processed)
                elif name == "MinMax":
                    processed = processing.apply_minmax_norm(processed)
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
