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

    def __init__(self, main_vm: MainViewModel, analysis_vm: AnalysisViewModel):
        super().__init__()
        self.main_vm = main_vm
        self.analysis_vm = analysis_vm # Need access to prep strategy
        self.service = LearningService()
        self.optimizer = OptimizationService()
        self.optimizer.log_message.connect(self.log_message.emit)
        
        # Threading state
        self.worker_thread = None
        self.worker = None
        
        self.opt_thread = None
        self.opt_worker = None
        
        # Smart Cache State
        self.last_train_config_hash = None
        self.cached_training_data = None # (X, y)
        self.current_pending_hash = None
        # Cache Handoff (From Optimization -> Training)
        self.cached_optimization_base_data = None
        
        # Cache Invalidation: When underlying data changes, clear caches
        self.analysis_vm.params_changed.connect(self._invalidate_base_cache)
        self.main_vm.refs_changed.connect(self._invalidate_base_cache)
        self.main_vm.files_changed.connect(self._invalidate_base_cache)
        self.main_vm.mode_changed.connect(self._invalidate_base_cache)

    def _invalidate_base_cache(self, *args):
        """Clear cached base data when settings change."""
        if self.cached_optimization_base_data is not None:
            self.cached_optimization_base_data = None
            self.log_message.emit("⚠️ Settings Changed: Base Data Cache Invalidated.")
        if self.cached_training_data is not None:
            self.cached_training_data = None
            self.last_train_config_hash = None

    def _ensure_ref_loaded(self):
        """Lazy Load Reference Data if not already cached."""
        from services.data_loader import load_hsi_data
        
        # White Ref
        if self.main_vm.cache_white is None and self.main_vm.white_ref:
            try:
                self.log_message.emit(f"Loading White Reference...")
                w_data, _ = load_hsi_data(self.main_vm.white_ref)
                if w_data is not None:
                    w_vec = np.nanmean(w_data.reshape(-1, w_data.shape[-1]), axis=0)
                    self.main_vm.cache_white = w_vec
            except Exception as e:
                self.log_message.emit(f"Warning: Failed to load White Ref: {e}")
                
        # Dark Ref
        if self.main_vm.cache_dark is None and self.main_vm.dark_ref:
            try:
                self.log_message.emit(f"Loading Dark Reference...")
                d_data, _ = load_hsi_data(self.main_vm.dark_ref)
                if d_data is not None:
                    d_vec = np.nanmean(d_data.reshape(-1, d_data.shape[-1]), axis=0)
                    self.main_vm.cache_dark = d_vec
            except Exception as e:
                self.log_message.emit(f"Warning: Failed to load Dark Ref: {e}")

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

    def run_training(self, output_path: str, n_features: int = 5, internal_sim: bool = False, silent: bool = False, model_type: str = "Linear SVM", test_ratio: float = 0.2):
        """
        Async Training Entry Point with Smart Caching.
        """
        if internal_sim:
             self.log_message.emit("Warning: internal_sim called on Async run_training.")
             return 0.0
             
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_message.emit("Training already running...")
            return

        # 1. Ensure Ref Data is Loaded (Lazy Load if needed)
        self._ensure_ref_loaded()
        
        # 2. State Snapshot
        vm_state = {
            'mode': self.analysis_vm.processing_mode,
            'threshold': self.analysis_vm.threshold,
            'mask_rules': self.analysis_vm.mask_rules,
            'prep_chain': self.analysis_vm.prep_chain,
            'white_ref': self.main_vm.cache_white, # Arrays
            'dark_ref': self.main_vm.cache_dark,
            'exclude_bands': self.analysis_vm.exclude_bands_str
        }
        
        # 2. Check Cache
        current_hash = self._compute_config_hash(vm_state)
        precomputed_data = None
        
        if self.cached_training_data is not None and current_hash == self.last_train_config_hash:
            if not silent: 
                self.log_message.emit("⚡ Smart Cache Hit! Reusing processed data (Instant Start)...")
            precomputed_data = self.cached_training_data
        else:
            self.current_pending_hash = current_hash # Store to update later
        
        # 3. Worker Params
        params = {
            'output_path': output_path,
            'n_features': n_features,
            'model_type': model_type,
            'test_ratio': test_ratio,
            'silent': silent,
            'base_data_cache': self.cached_optimization_base_data # Pass Handoff Data
        }
        
        # 4. Create Thread
        self.worker_thread = QThread()
        groups_copy = self.main_vm.file_groups.copy()
        
        self.worker = TrainingWorker(groups_copy, vm_state, self.main_vm.data_cache, params, precomputed_data=precomputed_data)
        self.worker.moveToThread(self.worker_thread)
        
        # 5. Connect Signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.progress_update)
        self.worker.log_message.connect(self.log_message)
        self.worker.training_finished.connect(self.on_worker_finished)
        self.worker.data_ready.connect(self.on_worker_data_ready) # Catch Base Cache
        self.worker.data_ready.connect(self.on_worker_data_ready) # Cache Update
        
        # Cleanup
        self.worker.training_finished.connect(self.worker.deleteLater)
        self.worker.training_finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_stopped)
        
        self.main_vm.request_save()
        self.worker_thread.start()
        
    def on_worker_data_ready(self, X, y):
        """
        Slot to receive Base Data from TrainingWorker.
        - Updates cached_training_data (for hash-based caching)
        - Updates cached_optimization_base_data (for handoff to Optimization)
        """
        # 1. Update Training Cache (Hash-based)
        self.cached_training_data = (X, y)
        if self.current_pending_hash:
            self.last_train_config_hash = self.current_pending_hash
        
        # 2. Update Handoff Cache (For Optimization reuse)
        self.cached_optimization_base_data = (X, y)
        self.log_message.emit("✅ Training Base Data Cached (Reusable for Optimization).")
            
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

    def run_optimization(self, output_path: str, model_type: str = "Linear SVM", test_ratio: float = 0.2):
        """
        Orchestrate Auto-ML Optimization (Background).
        """
        if self.analysis_vm is None: return
        
        if self.opt_thread and self.opt_thread.isRunning():
            self.log_message.emit("Optimization already running...")
            return

        # 0. Ensure Ref Data is Loaded
        self._ensure_ref_loaded()
        
        # 1. State Snapshot
        vm_state = {
            'mode': self.analysis_vm.processing_mode,
            'threshold': self.analysis_vm.threshold,
            'mask_rules': self.analysis_vm.mask_rules,
            'prep_chain': self.analysis_vm.prep_chain,
            'white_ref': self.main_vm.cache_white,
            'dark_ref': self.main_vm.cache_dark,
            'exclude_bands': self.analysis_vm.exclude_bands_str
        }
        
        # 2. Initial Params (From UI)
        prep_chain_copy = []
        for step in self.analysis_vm.prep_chain:
            new_step = {'name': step['name'], 'params': step['params'].copy()}
            prep_chain_copy.append(new_step)
            
        initial_params = {
            'prep': prep_chain_copy,
            'n_features': 5,
        }
        
        # 3. Create Worker & Thread
        self.opt_thread = QThread()
        groups_copy = self.main_vm.file_groups.copy()
        
        self.opt_worker = OptimizationWorker(
            groups_copy, 
            vm_state, 
            self.main_vm.data_cache, 
            initial_params,
            base_data_cache=self.cached_optimization_base_data
        )
        self.opt_worker.moveToThread(self.opt_thread)
        
        # 4. Connect Signals
        self.opt_thread.started.connect(self.opt_worker.run)
        self.opt_worker.log_message.connect(self.log_message)
        self.opt_worker.optimization_finished.connect(self.on_optimization_finished)
        
        # Cleanup
        self.opt_worker.optimization_finished.connect(self.opt_thread.quit)
        self.opt_worker.optimization_finished.connect(self.opt_worker.deleteLater)
        self.opt_thread.finished.connect(self.opt_thread.deleteLater)
        self.opt_thread.finished.connect(self._on_opt_thread_stopped)
        
        # 5. Start
        self.main_vm.request_save()
        self.opt_thread.start()
            
    def on_optimization_finished(self, success):
        # Notify View to Re-enable buttons
        self.training_finished.emit(success) 
        
        if success and self.opt_worker:
            # 1. Capture Best Params
            if hasattr(self.opt_worker, 'best_params'):
                best_params = self.opt_worker.best_params
                self.analysis_vm.set_preprocessing_chain(best_params['prep'])
            
            # 2. Capture Cached Base Data (Ref/Mask)
            # This allows TrainingWorker to skip file loading & masking
            if hasattr(self.opt_worker, 'cached_X') and self.opt_worker.cached_X is not None:
                self.cached_optimization_base_data = (self.opt_worker.cached_X, self.opt_worker.cached_y)
                self.log_message.emit("✅ Optimization Cache Saved for Training (Masked Data Valid).")

    def _on_opt_thread_stopped(self):
        self.opt_worker = None
        self.opt_thread = None
