from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication
from typing import List, Optional
import numpy as np
import os
import hashlib
import json

from viewmodels.main_vm import MainViewModel
from viewmodels.analysis_vm import AnalysisViewModel
from services.learning_service import LearningService
from services.data_loader import load_hsi_data
from services.band_selection_service import select_best_bands
from services.optimization_service import OptimizationService
from services.processing_service import ProcessingService
from services.training_worker import TrainingWorker
from services.optimization_worker import OptimizationWorker
from models import processing

class TrainingViewModel(QObject):
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    training_finished = pyqtSignal(bool) # Success?
    config_changed = pyqtSignal() # AI가 추가함: 설정 변경 시그널 (Auto-Save Trigger)

    def __init__(self, main_vm: MainViewModel, analysis_vm: AnalysisViewModel):
        super().__init__()
        self.main_vm = main_vm
        self.analysis_vm = analysis_vm # Need access to prep strategy
        self.service = LearningService()
        self.optimizer = OptimizationService()
        self.optimizer.log_message.connect(self.log_message.emit)
        
        # AI가 추가함: Training State Variables (Source of Truth)
        # 기본값은 default_project.json에서 덮어씌워지겠지만, 안전을 위해 초기화
        self.output_folder = "./output"
        self.model_name = "model"     # Default Filename
        self.model_desc = ""          # Default Description
        
        self.output_path = None       # Deprecated (Computed Property로 대체)
        
        self.model_type = "Linear SVM"
        self.val_ratio = 0.20
        self.n_features = 5
        
        # Threading state
        self.worker_thread = None
        self.worker = None
        
        self.opt_thread = None
        self.opt_worker = None
        
        # AI가 수정함: 캐시 구조 통합 - Base Data만 캐시 (전처리 전)
        # Dictionary Cache: {file_path: (X_base, y)} for selective training
        self.cached_base_data = {}  
        
        # AI가 추가함: 제외할 파일 목록
        self.excluded_files = set()
        
        # Cache Invalidation: When underlying data changes, clear caches
        # AI가 수정함: params_changed 대신 base_data_invalidated 연결 (Gap 변경 시 무효화 방지)
        self.analysis_vm.base_data_invalidated.connect(self._invalidate_base_cache)
        self.main_vm.refs_changed.connect(self._invalidate_base_cache)
        self.main_vm.files_changed.connect(self._invalidate_base_cache)
        self.main_vm.mode_changed.connect(self._invalidate_base_cache)

    @property
    def full_output_path(self):
        """Construct full path from folder and name"""
        import os
        filename = f"{self.model_name}.json"
        return os.path.join(self.output_folder, filename).replace("/", "\\")

    def get_config(self) -> dict:
        """AI가 추가함: 현재 Training 설정을 Dict로 반환 (저장용)"""
        return {
            "output_folder": self.output_folder,
            "model_name": self.model_name,
            "model_desc": self.model_desc,
            "model_type": self.model_type,
            "val_ratio": self.val_ratio,
            "n_features": self.n_features,
            # AI가 추가함: 제외 목록 저장 (list로 변환)
            "excluded_files": list(self.excluded_files)
        }

    def set_config(self, config: dict):
        """AI가 추가함: Dict 설정값을 받아 상태 업데이트 (로드용)"""
        if not config: return
        
        # 1. New Fields
        if "output_folder" in config:
            self.output_folder = config["output_folder"]
            self.model_name = config.get("model_name", "model")
            self.model_desc = config.get("model_desc", "")
        else:
            # 2. Backward Compatibility (Migration)
            old_path = config.get("output_path", "./output/model_config.json")
            if old_path:
                folder = os.path.dirname(old_path)
                filename = os.path.basename(old_path)
                name, _ = os.path.splitext(filename)
                
                self.output_folder = folder if folder else "./output"
                self.model_name = name if name else "model"
                self.model_desc = "" # New field default
        
        self.model_type = config.get("model_type", "Linear SVM")
        try:
            self.val_ratio = float(config.get("val_ratio", 0.20))
            self.n_features = int(config.get("n_features", 5))
        except ValueError:
            self.val_ratio = 0.20
            self.n_features = 5
            
        # AI가 추가함: 제외 목록 복원
        if "excluded_files" in config:
            self.excluded_files = set(config["excluded_files"])
        else:
            self.excluded_files = set()
            
        self.config_changed.emit()

    def set_file_excluded(self, path: str, excluded: bool):
        """AI가 추가함: 파일 제외 여부 토글 (캐시 유지)"""
        if excluded:
            self.excluded_files.add(path)
        else:
            self.excluded_files.discard(path)
        self.config_changed.emit()

    def _invalidate_base_cache(self, *args):
        """Clear cached base data when settings change."""
        # AI가 수정함: 캐시 통합으로 간소화 (Dict 초기화)
        if self.cached_base_data:
            self.cached_base_data = {}
            self.log_message.emit("⚠️ Settings Changed: Base Data Cache Cleared.")

    def _ensure_ref_loaded(self):
        """
        /// AI가 수정함: 중앙화된 메서드로 위임
        Lazy Load Reference Data if not already cached.
        """
        self.main_vm.ensure_refs_loaded()
    
    def _create_vm_state_snapshot(self):
        """
        AI가 수정함: Training/Optimization 공통 VM 상태 스냅샷 생성
        """
        return {
            'mode': self.analysis_vm.processing_mode,
            'threshold': self.analysis_vm.threshold,
            'mask_rules': self.analysis_vm.mask_rules,
            'prep_chain': self.analysis_vm.prep_chain,
            'white_ref': self.main_vm.cache_white,
            'dark_ref': self.main_vm.cache_dark,
            'exclude_bands': self.analysis_vm.exclude_bands_str
        }


    def _compute_config_hash(self, vm_state):
        """
        Compute hash of configuration that affects Data Processing.
        (Files + Mode + Threshold + Prep Chain + Mask Rules)
        Excludes: Model Type, n_features (these can change without re-processing)
        """
        # 1. Valid Files (Structure Aware)
        # We must include Group Names because changing a file's group changes its Label (y).
        file_structure = []
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        
        # Sort groups by name to ensure consistent order
        for name in sorted(self.main_vm.file_groups.keys()):
            if name.lower() not in EXCLUDED_NAMES:
                files = sorted(self.main_vm.file_groups[name])
                if files:
                    file_structure.append((name, files))
        
        # 2. Prep Params
        config = {
            'structure': file_structure, # Replaces flat file_list
            'mode': vm_state['mode'],
            'threshold': vm_state['threshold'],
            'mask': vm_state['mask_rules'],
            'prep': str(vm_state['prep_chain']),
            'white_path': self.main_vm.white_ref,
            'dark_path': self.main_vm.dark_ref,
            'exclude': vm_state['exclude_bands']
        }
        
        s = json.dumps(config, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()

    def run_training(self, output_path: Optional[str] = None, n_features: int = 0, internal_sim: bool = False, silent: bool = False, model_type: Optional[str] = None, test_ratio: float = 0.0):
        """
        Async Training Entry Point with Smart Caching.
        If args are provided, they override VM state (but VM state is preferred from UI).
        """
        # Fallback to VM State (Source of Truth)
        if output_path is None: output_path = self.full_output_path # AI가 수정함: computed property 사용
        if model_type is None: model_type = self.model_type
        if n_features <= 0: n_features = self.n_features
        if test_ratio <= 0.0: test_ratio = self.val_ratio

        if internal_sim:
             self.log_message.emit("Warning: internal_sim called on Async run_training.")
             return 0.0
        
        # AI가 수정함: Thread 안전 정리 강화
        if not self._safe_cleanup_thread():
            return  # 아직 실행 중

        # 1. Ensure Ref Data is Loaded (Lazy Load if needed)
        self._ensure_ref_loaded()
        
        # 2. State Snapshot (AI가 수정함: 공통 메서드 사용)
        vm_state = self._create_vm_state_snapshot()
        
        # AI가 수정함: 캐시 구조 통합 - Base Data 캐시만 사용
        # 3. Worker Params
        params = {
            'output_path': output_path,
            'n_features': n_features,
            'model_type': model_type,
            'test_ratio': test_ratio,
            'silent': silent,
            'base_data_cache': self.cached_base_data,  # 통합된 캐시
            # AI가 추가함: Naming Metadata
            'model_name': self.model_name,
            'model_desc': self.model_desc,
            # AI가 추가함: 제외 목록 전달
            'excluded_files': self.excluded_files.copy()
        }
        
        # 4. Create Thread
        self.worker_thread = QThread()
        groups_copy = self.main_vm.file_groups.copy()
        
        # AI가 수정함: precomputed_data 제거, base_data_cache만 사용
        self.worker = TrainingWorker(
            groups_copy, 
            vm_state, 
            self.main_vm.data_cache, 
            params,
            colors_map=self.main_vm.group_colors.copy() # AI가 수정함: 색상 전달
        )
        self.worker.moveToThread(self.worker_thread)
        
        # 5. Connect Signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.progress_update)
        self.worker.log_message.connect(self.log_message)
        self.worker.training_finished.connect(self._on_training_cleanup)  # AI가 수정함: cleanup 함수로 변경
        self.worker.base_data_ready.connect(self.on_base_data_ready)
        
        self.main_vm.request_save()
        self.worker_thread.start()
    
    def _safe_cleanup_thread(self):
        """
        AI가 수정함: Thread 안전 정리 - 실행 중이면 False, 아니면 정리 후 True
        """
        if self.worker_thread is None:
            return True
        
        if self.worker_thread.isRunning():
            self.log_message.emit("Training already running...")
            return False
        
        # Thread가 끝났지만 정리 안 됨 - 시그널 disconnect 후 정리
        try:
            self.worker_thread.started.disconnect()
            self.worker_thread.finished.disconnect()
        except (TypeError, RuntimeError):
            pass  # 이미 disconnect 됐거나 연결 없음
        
        if self.worker:
            try:
                self.worker.progress_update.disconnect()
                self.worker.log_message.disconnect()
                self.worker.training_finished.disconnect()
                self.worker.base_data_ready.disconnect()
            except (TypeError, RuntimeError):
                pass
        
        self.worker_thread = None
        self.worker = None
        return True
    
    def _on_training_cleanup(self, success):
        """
        AI가 수정함: training_finished 시그널 처리 + Thread 정리
        """
        # 1. 먼저 외부에 결과 알림
        self.training_finished.emit(success)
        
        # 2. Thread 정리
        if self.worker_thread:
            self.worker_thread.quit()
        
        # 3. 다음 이벤트 루프에서 참조 정리 (안전)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._on_thread_stopped)
        
    def on_base_data_ready(self, file_path, data_tuple):
        """
        AI가 수정함: 캐시 구조 통합 (Dict Update)
        Slot to receive Base Data (NO Preprocessing) for caching.
        """
        if file_path and data_tuple is not None:
            self.cached_base_data[file_path] = data_tuple
            # self.log_message.emit(f"✅ Cached data for {os.path.basename(file_path)}")
            
    def on_worker_finished(self, success):
        self.training_finished.emit(success)

    def _on_thread_stopped(self):
        self.worker = None
        self.worker_thread = None

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_message.emit("Stopping training...")
        if self.opt_worker:
            self.opt_worker.stop()
            self.log_message.emit("Stopping optimization...")

    def run_optimization(self, output_path: Optional[str] = None, model_type: Optional[str] = None, test_ratio: float = 0.0, n_features: int = 0):
        """
        Orchestrate Auto-ML Optimization (Background).
        Arguments are optional; if None, VM state is used.
        """
        # Fallback to VM State (Source of Truth)
        if output_path is None: output_path = self.full_output_path # AI가 수정함: computed property 사용
        if model_type is None: model_type = self.model_type
        if n_features <= 0: n_features = self.n_features
        # Optimizer typically uses internal splitting, but we respect UI choices where applicable.
        
        if self.analysis_vm is None: return
        
        # AI가 수정함: Thread 안전 정리 강화
        if not self._safe_cleanup_opt_thread():
            return  # 아직 실행 중

        # 0. Ensure Ref Data is Loaded
        self._ensure_ref_loaded()
        
        # 1. State Snapshot (AI가 수정함: 공통 메서드 사용)
        vm_state = self._create_vm_state_snapshot()
        
        # 2. Initial Params (From UI)
        prep_chain_copy = []
        for step in self.analysis_vm.prep_chain:
            new_step = {'name': step['name'], 'params': step['params'].copy()}
            prep_chain_copy.append(new_step)
            
        initial_params = {
            'prep': prep_chain_copy,
            'n_features': n_features, # AI가 수정함: UI 값 사용
            # AI가 추가함: 제외 목록 전달
            'excluded_files': self.excluded_files.copy()
        }
        
        # 3. Create Worker & Thread
        self.opt_thread = QThread()
        groups_copy = self.main_vm.file_groups.copy()
        
        # AI가 수정함: 통합된 캐시 사용
        self.opt_worker = OptimizationWorker(
            groups_copy, 
            vm_state, 
            self.main_vm.data_cache, 
            initial_params,
            model_type=model_type,  # AI가 수정함: 모델 타입 전달
            base_data_cache=self.cached_base_data
        )
        self.opt_worker.moveToThread(self.opt_thread)
        
        # 4. Connect Signals
        self.opt_thread.started.connect(self.opt_worker.run)
        self.opt_worker.log_message.connect(self.log_message)
        self.opt_worker.optimization_finished.connect(self._on_optimization_cleanup)  # AI가 수정함: cleanup 함수로 변경
        self.opt_worker.base_data_ready.connect(self.on_base_data_ready)
        
        # 5. Start
        self.main_vm.request_save()
        self.opt_thread.start()
    
    def _safe_cleanup_opt_thread(self):
        """
        AI가 수정함: Optimization Thread 안전 정리
        """
        if self.opt_thread is None:
            return True
        
        if self.opt_thread.isRunning():
            self.log_message.emit("Optimization already running...")
            return False
        
        # Thread가 끝났지만 정리 안 됨 - 시그널 disconnect 후 정리
        try:
            self.opt_thread.started.disconnect()
            self.opt_thread.finished.disconnect()
        except (TypeError, RuntimeError):
            pass
        
        if self.opt_worker:
            try:
                self.opt_worker.log_message.disconnect()
                self.opt_worker.optimization_finished.disconnect()
                self.opt_worker.base_data_ready.disconnect()
            except (TypeError, RuntimeError):
                pass
        
        self.opt_thread = None
        self.opt_worker = None
        return True
    
    def _on_optimization_cleanup(self, success):
        """
        AI가 수정함: optimization_finished 시그널 처리 + Thread 정리
        """
        # 1. 결과 처리 (best_params 저장)
        if success and self.opt_worker:
            if hasattr(self.opt_worker, 'best_params'):
                best_params = self.opt_worker.best_params
                self.analysis_vm.set_preprocessing_chain(best_params['prep'])
                self.best_n_features = best_params.get('n_features', 5)
            
            # AI가 수정함: 캐시 덮어쓰기 삭제 (Dict 유지)
            # OptimizationWorker는 파일별로 base_data_ready를 방출하므로 자동 캐싱됨.
            # 여기서 self.opt_worker.cached_X를 덮어쓰면 Tuple이 되어 구조가 망가짐.
        
        # 2. 외부에 결과 알림
        self.training_finished.emit(success)
        
        # 3. Thread 정리
        if self.opt_thread:
            self.opt_thread.quit()
        
        # 4. 다음 이벤트 루프에서 참조 정리
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._on_opt_thread_stopped)

    def _on_opt_thread_stopped(self):
        self.opt_worker = None
        self.opt_thread = None
