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
from services.experiment_worker import ExperimentWorker  # AI가 수정함: Experiment Grid 워커 임포트
from models import processing

_ALL_BAND_METHODS = ['spa', 'full', 'anova_f', 'spa_lda_fast', 'lda_coef']  # AI가 수정함: spa_lda_greedy 제외 — Optimize 루프에서 cv×k×B 반복으로 매우 느림 (Experiment에서만 선택적 사용)

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
        self.band_selection_method = "spa"  # AI가 수정함: 밴드 선택 방법 (기본값 SPA)
        self.experiment_band_methods = ["spa"]  # AI가 수정함: Export Matrix용 다중 밴드 선택 기본값
        self.experiment_model_types = ["Linear SVM"]  # AI가 수정함: Export Matrix용 다중 모델 기본값
        self.experiment_n_bands_list: list = [self.n_features]  # AI가 추가함: 실험 그리드 n_bands 목록 (기본: 현재 n_features)
        self.experiment_gap_min: int = 1   # AI가 추가함: 실험 그리드 gap 범위 최솟값
        self.experiment_gap_max: int = 20  # AI가 추가함: 실험 그리드 gap 범위 최댓값
        
        # Threading state
        self.worker_thread = None
        self.worker = None
        
        self.opt_thread = None
        self.opt_worker = None
        
        self.exp_thread = None   # AI가 수정함: Experiment Grid 스레드
        self.exp_worker = None   # AI가 수정함: Experiment Grid 워커
        
        # AI가 수정함: 캐시 구조 통합 - Base Data만 캐시 (전처리 전)
        # Dictionary Cache: {file_path: (X_base, y)} for selective training
        self.cached_base_data = {}  
        
        # AI가 추가함: 제외할 파일 목록
        self.excluded_files = set()
        
        # AI가 추가함: 최적화 결과 저장 (최적화 완료 전 UI 접근 시 AttributeError 방지)
        self.best_n_features = self.n_features
        self.best_band_method: Optional[str] = None  # AI가 추가함: 최적화 완료 후 설정되는 최적 밴드 선택 방법 (transient)
        
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
            "band_selection_method": self.band_selection_method,  # AI가 수정함: 밴드 선택 방법 저장
            "experiment_band_methods": list(self.experiment_band_methods),  # AI가 수정함: Export Matrix 밴드 방법 목록 저장
            "experiment_model_types": list(self.experiment_model_types),  # AI가 수정함: Export Matrix 모델 목록 저장
            "experiment_n_bands_list": list(self.experiment_n_bands_list),  # AI가 추가함: 실험 그리드 n_bands 목록 저장
            "experiment_gap_min": self.experiment_gap_min,  # AI가 추가함: 실험 그리드 gap 최솟값 저장
            "experiment_gap_max": self.experiment_gap_max,  # AI가 추가함: 실험 그리드 gap 최댓값 저장
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
        self.band_selection_method = config.get("band_selection_method", "spa")  # AI가 수정함: backward compat
        # AI가 수정함: Variance→Full Band 마이그레이션 (기존 프로젝트 파일 호환)
        if self.band_selection_method == "variance":  # AI가 수정함:
            self.band_selection_method = "full"  # AI가 수정함:
        # AI가 수정함: 알 수 없는 method string 방어 (프로젝트 파일 호환)
        _known_methods = {'spa', 'full', 'anova_f', 'spa_lda_fast', 'spa_lda_greedy', 'lda_coef'}
        if self.band_selection_method not in _known_methods:  # AI가 수정함:
            self.band_selection_method = 'spa'  # AI가 수정함: unknown method → spa fallback
        _raw_exp_band_methods = config.get("experiment_band_methods", [self.band_selection_method])  # AI가 수정함: 다중 밴드 방법 로드
        if isinstance(_raw_exp_band_methods, str):  # AI가 수정함: 구버전/수동 편집 호환
            _raw_exp_band_methods = [_raw_exp_band_methods]  # AI가 수정함: 문자열 → 단일 리스트 정규화
        self.experiment_band_methods = [m for m in _raw_exp_band_methods if m in _known_methods]  # AI가 수정함: 허용된 밴드 방법만 유지
        if not self.experiment_band_methods:  # AI가 수정함: 비어 있으면 현재 단일 선택으로 fallback
            self.experiment_band_methods = [self.band_selection_method]  # AI가 수정함: 최소 1개 보장
        _known_model_types = {"Linear SVM", "PLS-DA", "LDA", "Ridge Classifier", "Logistic Regression"}  # AI가 수정함: Export Matrix 허용 모델 목록
        _raw_exp_model_types = config.get("experiment_model_types", [self.model_type])  # AI가 수정함: 다중 모델 로드
        if isinstance(_raw_exp_model_types, str):  # AI가 수정함: 구버전/수동 편집 호환
            _raw_exp_model_types = [_raw_exp_model_types]  # AI가 수정함: 문자열 → 단일 리스트 정규화
        self.experiment_model_types = [m for m in _raw_exp_model_types if m in _known_model_types]  # AI가 수정함: 허용된 모델만 유지
        if not self.experiment_model_types:  # AI가 수정함: 비어 있으면 현재 단일 모델로 fallback
            self.experiment_model_types = [self.model_type]  # AI가 수정함: 최소 1개 보장
        # AI가 추가함: 실험 그리드 n_bands 목록 / gap 범위 로드
        _raw_n_bands_list = config.get('experiment_n_bands_list', [self.n_features])
        if isinstance(_raw_n_bands_list, list) and _raw_n_bands_list:
            self.experiment_n_bands_list = [int(v) for v in _raw_n_bands_list if str(v).isdigit() or isinstance(v, int)]
        if not self.experiment_n_bands_list:
            self.experiment_n_bands_list = [self.n_features]
        try:
            self.experiment_gap_min = int(config.get('experiment_gap_min', 1))
            self.experiment_gap_max = int(config.get('experiment_gap_max', 20))
        except (ValueError, TypeError):
            self.experiment_gap_min = 1
            self.experiment_gap_max = 20
            
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

    def _get_raw_band_count_cached_only(self):
        """
        /// <ai>AI가 작성함</ai>
        Return RAW sensor band count using only already-cached files to avoid blocking
        the main (UI) thread with disk I/O.

        Falls back to 0 if no cached files are available — OptimizationWorker.run()
        will set self.raw_band_count from cached_X.shape[1] after _prepare_base_data().
        """
        EXCLUDED_NAMES = {"-", "unassigned", "trash", "ignore"}
        for group_name, files in sorted(self.main_vm.file_groups.items()):
            if group_name.lower() in EXCLUDED_NAMES:
                continue
            for path in sorted(files):
                if path in self.excluded_files:
                    continue
                if path in self.main_vm.data_cache:
                    cube, waves = self.main_vm.data_cache[path]
                    count = int(cube.shape[2])
                    if waves is not None and len(waves) > 0:
                        count = int(len(waves))
                    return count  # AI가 수정함: 첫 번째 캐시 히트만으로도 충분
        return 0  # AI가 수정함: 캐시 미스 — 0 반환하여 워커가 lazy 결정하도록 위임

    def _get_raw_band_count(self):
        """Return authoritative RAW sensor band count from loaded cubes/wavelengths."""
        EXCLUDED_NAMES = {"-", "unassigned", "trash", "ignore"}
        candidate_files = []
        for group_name, files in sorted(self.main_vm.file_groups.items()):
            if group_name.lower() in EXCLUDED_NAMES:
                continue
            for path in sorted(files):
                if path not in self.excluded_files:
                    candidate_files.append(path)

        if not candidate_files:
            raise ValueError("No active files available to determine raw band count.")

        raw_band_count = None
        for file_path in candidate_files:
            if file_path in self.main_vm.data_cache:
                cube, waves = self.main_vm.data_cache[file_path]
            else:
                cube, waves = load_hsi_data(file_path)
                cube = np.nan_to_num(cube)
                self.main_vm.data_cache[file_path] = (cube, waves)

            current_count = int(cube.shape[2])
            if waves is not None and len(waves) > 0:
                current_count = int(len(waves))

            if raw_band_count is None:
                raw_band_count = current_count
            elif raw_band_count != current_count:
                raise ValueError(
                    f"Inconsistent raw band count across files: {raw_band_count} != {current_count} ({file_path})"
                )

        if raw_band_count is None or raw_band_count <= 0:
            raise ValueError("Failed to determine raw band count from active files.")

        return raw_band_count


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
        raw_band_count = self._get_raw_band_count()
        
        # AI가 수정함: 캐시 구조 통합 - Base Data 캐시만 사용
        # 3. Worker Params
        params = {
            'output_path': output_path,
            'n_features': n_features,
            'model_type': model_type,
            'test_ratio': test_ratio,
            'silent': silent,
            'base_data_cache': dict(self.cached_base_data),  # AI가 수정함: race condition 방지 — snapshot copy 전달
            # AI가 추가함: Naming Metadata
            'model_name': self.model_name,
            'model_desc': self.model_desc,
            # AI가 추가함: 제외 목록 전달
            'excluded_files': self.excluded_files.copy(),
            'band_selection_method': self.band_selection_method,  # AI가 수정함: 하드코딩 제거
            'raw_band_count': raw_band_count,
        }
        
        # 4. Create Thread
        self.worker_thread = QThread()
        # AI가 수정함: shallow copy → deep copy — 훈련 중 메인 스레드의 file_groups 수정 시 race condition 방지
        groups_copy = {k: list(v) for k, v in self.main_vm.file_groups.items()}
        
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
            # AI가 수정함: V2-DLT — Qt 객체 결정론적 해제 (deleteLater 패턴)
            if self.worker:
                self.worker_thread.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
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
        if self.exp_worker:  # AI가 수정함: experiment worker 중단
            self.exp_worker.stop()
            self.log_message.emit("Stopping experiment...")

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
        raw_band_count = self._get_raw_band_count_cached_only()  # AI가 수정함: 메인 스레드 I/O 블로킹 방지 — 캐시에 있는 파일만 사용, 없으면 0 전달하여 워커에서 lazy 결정
        
        # 2. Initial Params (From UI)
        prep_chain_copy = []
        for step in self.analysis_vm.prep_chain:
            new_step = {'name': step['name'], 'params': step['params'].copy()}
            prep_chain_copy.append(new_step)
            
        initial_params = {
            'prep': prep_chain_copy,
            'n_features': n_features, # AI가 수정함: UI 값 사용
            # AI가 추가함: 제외 목록 전달
            'excluded_files': self.excluded_files.copy(),
            'band_selection_method': self.band_selection_method,  # AI가 수정함: 하드코딩 제거
            'raw_band_count': raw_band_count,
        }
        
        # 3. Create Worker & Thread
        self.opt_thread = QThread()
        # AI가 수정함: shallow copy → deep copy — 최적화 중 메인 스레드의 file_groups 수정 시 race condition 방지
        groups_copy = {k: list(v) for k, v in self.main_vm.file_groups.items()}
        
        # AI가 수정함: 통합된 캐시 사용
        self.opt_worker = OptimizationWorker(
            groups_copy, 
            vm_state, 
            self.main_vm.data_cache, 
            initial_params,
            model_type=model_type,  # AI가 수정함: 모델 타입 전달
            base_data_cache=dict(self.cached_base_data),  # AI가 수정함: race condition 방지 — snapshot copy 전달
            output_dir=self.output_folder,  # AI가 수정함: output_dir 전달
            band_methods=_ALL_BAND_METHODS  # AI가 추가함: 전체 밴드 선택 방법 목록 전달
        )
        self.opt_worker.moveToThread(self.opt_thread)
        
        # 4. Connect Signals
        self.opt_thread.started.connect(self.opt_worker.run)
        self.opt_worker.log_message.connect(self.log_message)
        self.opt_worker.progress_update.connect(self.progress_update)  # AI가 수정함: 최적화 진행률 UI 연결 누락 수정
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
                self.opt_worker.progress_update.disconnect()  # AI가 수정함: progress_update 연결 추가에 따른 cleanup 대응
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
                if best_params.get('band_selection_method') != 'full':  # AI가 수정함: Full Band는 스피너 업데이트 불필요
                    self.best_n_features = best_params.get('n_features', 5)  # AI가 수정함: SPA 등 일반 모드만 업데이트
                if best_params.get('band_selection_method'):  # AI가 추가함: 최적 밴드 선택 방법 저장 (transient)
                    self.best_band_method = best_params['band_selection_method']
            
            # AI가 수정함: 캐시 덮어쓰기 삭제 (Dict 유지)
            # OptimizationWorker는 파일별로 base_data_ready를 방출하므로 자동 캐싱됨.
            # 여기서 self.opt_worker.cached_X를 덮어쓰면 Tuple이 되어 구조가 망가짐.
        
        # 2. 외부에 결과 알림
        self.training_finished.emit(success)
        
        # 3. Thread 정리
        if self.opt_thread:
            # AI가 수정함: V2-DLT — Qt 객체 결정론적 해제 (deleteLater 패턴)
            if self.opt_worker:
                self.opt_thread.finished.connect(self.opt_worker.deleteLater)
            self.opt_thread.finished.connect(self.opt_thread.deleteLater)
            self.opt_thread.quit()
        
        # 4. 다음 이벤트 루프에서 참조 정리
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._on_opt_thread_stopped)

    def _on_opt_thread_stopped(self):
        self.opt_worker = None
        self.opt_thread = None

    def run_experiment_grid(self, band_methods: list, model_types: list,
                            n_bands_list: Optional[list] = None, gap_min: int = 1, gap_max: int = 20):  # AI가 수정함: 4D 실험 검색 공간 파라미터 추가
        """Async Experiment Grid Entry Point."""
        _n_bands_list = n_bands_list if n_bands_list is not None else [self.n_features]  # AI가 추가함: 기본값 현재 n_features
        if not self._safe_cleanup_exp_thread(): return
        self.experiment_band_methods = list(band_methods)  # AI가 수정함: Export Matrix 선택 상태 동기화
        self.experiment_model_types = list(model_types)  # AI가 수정함: Export Matrix 선택 상태 동기화
        self._ensure_ref_loaded()
        vm_state = self._create_vm_state_snapshot()
        raw_band_count = self._get_raw_band_count()
        params = {
            'band_methods': band_methods,
            'model_types': model_types,
            'n_bands_list': _n_bands_list,  # AI가 수정함: 단일 n_bands → n_bands_list로 교체
            'gap_range': (gap_min, gap_max),  # AI가 추가함: gap 범위 전달
            'test_ratio': self.val_ratio,
            'output_dir': self.output_folder,
            'excluded_files': self.excluded_files.copy(),
            'band_selection_method': self.band_selection_method,
            'raw_band_count': raw_band_count,
        }
        self.exp_thread = QThread()
        groups_copy = {k: list(v) for k, v in self.main_vm.file_groups.items()}
        self.exp_worker = ExperimentWorker(
            groups_copy, vm_state, self.main_vm.data_cache, params,
            base_data_cache=dict(self.cached_base_data)
        )
        self.exp_worker.moveToThread(self.exp_thread)
        self.exp_thread.started.connect(self.exp_worker.run)
        self.exp_worker.log_message.connect(self.log_message)
        self.exp_worker.progress_update.connect(self.progress_update)
        self.exp_worker.experiment_finished.connect(self._on_experiment_cleanup)
        self.exp_worker.base_data_ready.connect(self.on_base_data_ready)
        self.main_vm.request_save()
        self.exp_thread.start()

    def _safe_cleanup_exp_thread(self):  # AI가 수정함: Experiment thread 안전 정리
        if self.exp_thread is None: return True
        if self.exp_thread.isRunning():
            self.log_message.emit("Experiment already running...")
            return False
        try:
            self.exp_thread.started.disconnect()
            self.exp_thread.finished.disconnect()
        except (TypeError, RuntimeError): pass
        if self.exp_worker:
            try:
                self.exp_worker.log_message.disconnect()
                self.exp_worker.experiment_finished.disconnect()
                self.exp_worker.base_data_ready.disconnect()
            except (TypeError, RuntimeError): pass
        self.exp_thread = None
        self.exp_worker = None
        return True

    def _on_experiment_cleanup(self, success):  # AI가 수정함: experiment thread cleanup
        self.training_finished.emit(success)
        if self.exp_thread:
            if self.exp_worker:
                self.exp_thread.finished.connect(self.exp_worker.deleteLater)
            self.exp_thread.finished.connect(self.exp_thread.deleteLater)
            self.exp_thread.quit()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._on_exp_thread_stopped)

    def _on_exp_thread_stopped(self):  # AI가 수정함: experiment thread reference cleanup
        self.exp_worker = None
        self.exp_thread = None
