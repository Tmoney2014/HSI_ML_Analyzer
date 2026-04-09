from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

from services.data_loader import load_hsi_data
from services.learning_service import LearningService
from services.processing_service import ProcessingService
from services.band_selection_service import select_best_bands
from services.optimization_service import OptimizationService

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
    
    def __init__(self, file_groups, vm_state, main_vm_cache, initial_params, model_type="Linear SVM", base_data_cache=None, output_dir=None):  # AI가 수정함: output_dir 인자 추가
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
        self.output_dir = output_dir  # AI가 수정함: 저장 경로
        
        # AI가 추가함: 제외 파일 목록
        self.excluded_files = initial_params.get('excluded_files', set())
        
        self.model_type = model_type
        self.band_selection_method = initial_params.get('band_selection_method', 'spa')  # AI가 수정함: 하드코딩 제거
        
        # Thread-safe Cache will be created in run() to ensure thread affinity
        self.service = None 
        
        self.is_running = True
        self._total_trials = 0  # AI가 수정함: 진행률 추적용 총 trial 수
        self._completed_trials = 0  # AI가 수정함: 완료된 trial 수
        
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

            assert self.cached_X is not None and self.cached_y is not None  # AI가 수정함: 캐시 준비 후 None 방지
            self.log_message.emit(f"Data Ready: {self.cached_X.shape[0]} pixels cached in memory.")
            # AI가 수정함: 총 trial 수 사전 계산 (progress bar용)
            _band_range = [self.initial_params.get('n_features', 5)]  # AI가 수정함: Full Band 기본 trial 수 1개
            if self.initial_params.get('band_selection_method') != 'full':  # AI가 수정함: 일반 모드 trial 범위 사용
                _band_range = list(range(5, 41, 5))  # AI가 수정함: 밴드 수 탐색 범위
            _gap_range = [0]  # AI가 수정함: SimpleDeriv 없을 때 gap trial 없음
            if any(s.get('name') == 'SimpleDeriv' for s in self.initial_params.get('prep', [])):  # AI가 수정함: SimpleDeriv 포함 여부 확인
                _gap_range = list(range(1, 41))  # AI가 수정함: gap 탐색 범위
            self._total_trials = len(_band_range) * len(_gap_range)  # AI가 수정함: progress bar 분모 계산
            self._completed_trials = 0  # AI가 수정함: 진행률 카운터 초기화
            
            # 2. Instantiate Service in THIS thread
            self.service = OptimizationService()
            self.service.log_message.connect(self.log_message.emit)
            
            # Run the generic optimization algorithm
            best_params, history = self.service.run_optimization(self.initial_params, self.trial_callback)

            actual_best_acc = max((acc for _, acc in history), default=0.0)  # AI가 수정함: 마지막 trial이 아닌 실제 최고 정확도 계산
            if self.output_dir:  # AI가 수정함: output_dir이 있을 때만 report 저장
                self.service._generate_report(best_params, actual_best_acc, history, output_dir=self.output_dir)  # AI가 수정함: CSV/JSON 저장
            self.log_message.emit(f"=== Optimization Finished. Best Accuracy: {actual_best_acc:.2f}% ===")  # AI가 수정함: max() 기반 best accuracy 사용
            
            self.best_params = best_params
            
            # AI가 수정함: 통합 캐시 Update는 Load Loop 내부에서 개별적으로 수행됨
            # self.cached_X / cached_y는 내부 최적화용으로만 유지
            
            self.progress_update.emit(100)  # AI가 수정함: 완료 시 100% emit
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
        self._completed_trials += 1  # AI가 수정함: 진행률 추적
        if self._total_trials > 0:  # AI가 수정함: 0 나누기 방어
            pct = int(self._completed_trials * 100 / self._total_trials)  # AI가 수정함: 진행률 계산
            self.progress_update.emit(min(pct, 99))  # AI가 수정함: 100%는 완료 시에만
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
                             # AI가 수정함: nan_to_num 이중 호출 제거 — O-1. 캐시 저장 전 1회만 처리.
                             cube = np.nan_to_num(cube)
                             self.data_cache[f] = (cube, waves)
                        
                        # Use Service to get valid pixels with Ref Conversion
                        # AI가 수정함: process_cube(prep_chain=[]) 대신 get_base_data() 사용 (training_worker와 동일 경로)
                        data, mask = ProcessingService.get_base_data(
                            cube,
                            mode=self.processing_mode,
                            threshold=self.threshold,
                            mask_rules=self.mask_rules,
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
        assert self.cached_X is not None and self.cached_y is not None  # AI가 수정함: 평가 전 캐시 None 방지
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

        # AI가 추가함: Gap/SG 기반 SPA 상한 자동 제외
        # SimpleDeriv 사용 시: 전처리 후 자연히 상한 제한됨. SG radius만 추가 제약.
        # Absorbance 모드 + SimpleDeriv 없음: C# LogGapFeatureExtractor가 gap offset으로 밴드 접근
        _sg_radius = 0
        for _step in prep_chain:
            _nm = _step.get('name', '')
            if _nm == 'SG': _sg_radius = _step.get('params', {}).get('win', 5) // 2

        _upper_offset = _sg_radius
        if _upper_offset > 0:
            _upper_limit = n_bands - _upper_offset
            if 0 < _upper_limit < n_bands:
                exclude_indices = list(set(exclude_indices) | set(range(_upper_limit, n_bands)))

        # Use Full Data for SPA (User Request: No Downsampling)
        dummy = X.reshape(-1, 1, X.shape[1])
        
        selected_indices, _, _ = select_best_bands(
            dummy, 
            n_bands=n_features, 
            method=self.band_selection_method,  # AI가 수정함: 하드코딩 제거
            exclude_indices=exclude_indices,
            labels=self.cached_y  # AI가 수정함: supervised 밴드 선택 방법을 위해 cached_y 전달
        )
        
        # 3. Train
        X_sub = X[:, selected_indices]
        
        try:
            learning = LearningService()
            # If dataset is HUGE, this line will be the bottleneck in loop.
            model, metrics = learning.train_model(X_sub, y, model_type=self.model_type, test_ratio=0.2)
            # metrics['TestAccuracy'] is already 0~100 scale float
            return metrics['TestAccuracy']
        except Exception as e:
            # AI가 수정함: bare except → Exception으로 변경, 실제 에러를 로그로 출력
            self.log_message.emit(f"[Optimization Trial Error] {type(e).__name__}: {e}")
            return 0.0

    def stop(self):
        self.is_running = False
