from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

from services.data_loader import load_hsi_data
from services.learning_service import LearningService
from services.processing_service import ProcessingService
from services.band_selection_service import select_best_bands
from services.optimization_service import OptimizationService
from models import processing

class OptimizationWorker(QObject):
    """
    Worker to run Auto-ML Optimization in a Background Thread.
    Decoupled from ViewModels to ensure Thread Safety.
    Implements 'Pre-loading' strategy to speed up optimization loop.
    """
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    optimization_finished = pyqtSignal(bool) # Success
    data_ready = pyqtSignal(object, object)  # AI가 수정함: 캐시 저장용 (X, y)
    base_data_ready = pyqtSignal(object, object)  # AI가 수정함: 통합 캐시용 Base Data
    
    def __init__(self, file_groups, vm_state, main_vm_cache, initial_params, model_type="Linear SVM", base_data_cache=None):
        super().__init__()
        self.file_groups = file_groups
        # VM State Snapshot (Read-Only)
        self.processing_mode = vm_state['mode']
        self.threshold = vm_state['threshold']
        self.mask_rules = vm_state['mask_rules']
        # Init with empty prep first, will be tuned
        self.base_prep_chain = vm_state['prep_chain'] 
        
        # Reference Data
        self.white_ref = vm_state.get('white_ref')
        self.dark_ref = vm_state.get('dark_ref')
        self.exclude_bands_str = vm_state.get('exclude_bands', "")
        
        self.data_cache = main_vm_cache 
        self.initial_params = initial_params
        self.base_data_cache = base_data_cache # Handoff Data
        
        # AI가 추가함: 제외 파일 목록
        self.excluded_files = initial_params.get('excluded_files', set())
        
        self.model_type = model_type
        
        # Thread-safe Cache will be created in run() to ensure thread affinity
        self.service = None 
        
        self.is_running = True
        
        # Cache for Pre-loaded Validation Data (Raw/Reflectance before Prep)
        self.cached_X = None
        self.cached_y = None

    def run(self):
        try:
            self.log_message.emit("=== Starting Auto-Optimization (Background) ===")
            
            # 1. Prepare Data ONCE (Speed Optimization)
            self.log_message.emit("Pre-loading and masking data (Full Dataset)...")
            if not self._prepare_base_data():
                self.log_message.emit("Error: No valid data found for optimization.")
                self.optimization_finished.emit(False)
                return

            self.log_message.emit(f"Data Ready: {self.cached_X.shape[0]} pixels cached in memory.")
            
            # 2. Instantiate Service in THIS thread
            self.service = OptimizationService()
            self.service.log_message.connect(self.log_message.emit)
            
            # Run the generic optimization algorithm
            best_params, history = self.service.run_optimization(self.initial_params, self.trial_callback)
            
            self.log_message.emit(f"=== Optimization Finished. Best Accuracy: {history[-1][1]:.2f}% ===")
            
            self.best_params = best_params
            
            # AI가 수정함: 통합 캐시 Update는 Load Loop 내부에서 개별적으로 수행됨
            # self.cached_X / cached_y는 내부 최적화용으로만 유지
            
            self.optimization_finished.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"Optimization Error: {e}")
            import traceback
            # self.log_message.emit(traceback.format_exc())
            self.optimization_finished.emit(False)

    def trial_callback(self, params):
        if not self.is_running: raise Exception("Optimization Stopped")
        
        # 1. Parse Trial Params
        trial_prep = params['prep']
        n_features = params.get('n_features', 5)
        
        # 2. Run Pipeline on Cached Data (Fast)
        score = self._evaluate_cached_data(trial_prep, n_features)
        return score

    def _prepare_base_data(self):
        """
        Loads all files, applies Masking, and caches the Raw/Pre-Ref Valid Pixels.
        NO Subsampling Limit (User Request).
        """
        # 0. Handoff Cache Check (Bi-Directional)
        # 0. Handoff Cache Check (Dict Cache Awareness)
        # base_data_cache is now Dict {path: (X_b, y)}
        if self.base_data_cache is None: self.base_data_cache = {}

        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        valid_groups = {}
        for name, files in self.file_groups.items():
            if len(files) > 0 and name.lower() not in EXCLUDED_NAMES:
                valid_groups[name] = files
        
        valid_groups = dict(sorted(valid_groups.items()))  # Ensure deterministic order
        
        if len(valid_groups) < 2: return False

        X_all = []
        y_all = []
        
        X_all = []
        y_all = []
        
        # Load EVERYTHING (Selective)
        total_classes = len(valid_groups)
        for label_id, (class_name, files) in enumerate(valid_groups.items()):
            class_pixels = 0  # AI가 수정함: 클래스별 픽셀 수 추적
            # AI가 수정함: 파일 로딩 순서 고정 (재현성 보장)
            sorted_files = sorted(files)
            for f in sorted_files:
                if not self.is_running: return False
                
                # 1. Check Exclusion
                if f in self.excluded_files:
                    # self.log_message.emit(f"   [Skip] '{os.path.basename(f)}'")
                    continue

                try:
                    # 2. Check Base Data Cache (Dict)
                    if f in self.base_data_cache:
                        # self.log_message.emit(f"   [Cache] '{os.path.basename(f)}'")
                        data, _ = self.base_data_cache[f] # (X_base, y)
                        
                        # OptimizationWorker expects "Base Data" (Masked + Reflectance)
                        # So we can use it directly.
                        if data.shape[0] > 0:
                            X_all.append(data)
                            y_all.append(np.full(data.shape[0], label_id))
                            class_pixels += data.shape[0]
                    else:
                        # 3. Load from Disk
                        # Cache Check (File)
                        if f in self.data_cache:
                             cube, waves = self.data_cache[f]
                        else:
                             cube, waves = load_hsi_data(f)
                             cube = np.nan_to_num(cube) 
                             self.data_cache[f] = (cube, waves)
                        
                        cube = np.nan_to_num(cube)
                        
                        # Use Service to get valid pixels with Ref Conversion
                        data, mask = ProcessingService.process_cube(
                            cube,
                            mode=self.processing_mode,
                            threshold=self.threshold,
                            mask_rules=self.mask_rules,
                            prep_chain=[], 
                            white_ref=self.white_ref,
                            dark_ref=self.dark_ref
                        )
                        
                        if data.shape[0] > 0:
                            X_all.append(data)
                            y_all.append(np.full(data.shape[0], label_id))
                            class_pixels += data.shape[0]
                            
                            # Update Base Cache (Emit)
                            # Emit format: (file_path, (data, None))
                            self.base_data_ready.emit(f, (data, None))
                            
                except Exception as e:
                    # AI가 수정함: Strict Mode - 로드 에러 은폐 금지
                    self.log_message.emit(f"Critical Error loading {f}: {e}")
                    return False
            
            # AI가 수정함: 클래스별 로딩 완료 로그
            self.log_message.emit(f"  [{label_id+1}/{total_classes}] {class_name}: {class_pixels:,} pixels")
        
        if not X_all: return False
        
        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        
        # NO LIMIT - Use All Data
        self.cached_X = X
        self.cached_y = y
            
        return True

    def _evaluate_cached_data(self, prep_chain, n_features):
        """
        Apply Preprocessing to Cached Data -> Band Selection -> Train -> Score
        """
        X = self.cached_X.copy() # (N, Bands)
        y = self.cached_y
        
        # 1. Apply Preprocessing chain manually
        from services.processing_service import ProcessingService
        
        # Apply Prep Steps
        X = ProcessingService.apply_preprocessing_chain(X, prep_chain)
            
        # 2. Band Selection (SPA)
        # Parse Exclude Bands (From self.exclude_bands_str)
        exclude_indices = []
        if self.exclude_bands_str:
            try:
                for part in self.exclude_bands_str.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        exclude_indices.extend(range(start - 1, end))
                    else:
                        exclude_indices.append(int(part) - 1)
            except: pass
        
        # AI가 수정함: Gap Diff로 밴드 수가 줄어들면 범위 초과 인덱스 무시
        n_bands = X.shape[1]
        exclude_indices = [i for i in exclude_indices if 0 <= i < n_bands]

        # Use Full Data for SPA (User Request: No Downsampling)
        dummy = X.reshape(-1, 1, X.shape[1])
        
        selected_indices, _, _ = select_best_bands(
            dummy, 
            n_bands=n_features, 
            method='spa',
            exclude_indices=exclude_indices
        )
        
        # 3. Train
        X_sub = X[:, selected_indices]
        
        try:
            learning = LearningService()
            # If dataset is HUGE, this line will be the bottleneck in loop.
            model, metrics = learning.train_model(X_sub, y, model_type=self.model_type, test_ratio=0.2)
            # metrics['TestAccuracy'] is already 0~100 scale float
            return metrics['TestAccuracy']
        except:
            return 0.0

    def stop(self):
        self.is_running = False
