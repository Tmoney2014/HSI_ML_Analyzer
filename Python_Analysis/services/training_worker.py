from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from services.data_loader import load_hsi_data
from services.learning_service import LearningService
from services.processing_service import ProcessingService
from services.band_selection_service import select_best_bands
import os

class TrainingWorker(QObject):
    """
    Worker class to run training in a background thread.
    Refactored for Robustness & Smart Caching (No Pixel Limits).
    """
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    training_finished = pyqtSignal(bool)
    data_ready = pyqtSignal(object, object) # Emits (X, y) for caching
    
    def __init__(self, file_groups, analysis_vm_state, main_vm_cache, params, precomputed_data=None):
        super().__init__()
        self.file_groups = file_groups
        # Snapshot of state
        self.processing_mode = analysis_vm_state['mode']
        self.threshold = analysis_vm_state['threshold']
        self.mask_rules = analysis_vm_state['mask_rules']
        self.prep_chain = analysis_vm_state['prep_chain']
        
        # Reference Data (Arrays)
        self.white_ref = analysis_vm_state.get('white_ref')
        self.dark_ref = analysis_vm_state.get('dark_ref')
        
        self.exclude_bands_str = analysis_vm_state.get('exclude_bands', "")
        
        # Thread-safe Cache
        self.data_cache = main_vm_cache 
        
        self.params = params 
        self.precomputed_data = precomputed_data # (X, y) Fully Processed
        self.base_data_cache = params.get('base_data_cache') # (X, y) Ref/Masked Only
        self.is_running = True

    def run(self):
        try:
            # 1. Setup & Validation
            if not self._validate_inputs(): return

            output_path = self.params['output_path']
            n_features = self.params['n_features']
            model_type = self.params['model_type']
            test_ratio = self.params['test_ratio']
            silent = self.params.get('silent', False)
            
            # Log Params
            self._log_configuration(silent, n_features)

            # 2. Data Acquisition (Smart Cache Check)
            self.progress_update.emit(0)
            X, y, label_map = self._get_data_with_caching(silent)
            
            if X is None or not self.is_running: return # Error or Stopped

            # Cache Emit (If new data was processed)
            if self.precomputed_data is None:
                self.data_ready.emit(X, y)

            self.progress_update.emit(60)

            # 3. Band Selection
            if not silent: self.log_message.emit(f"Selecting best {n_features} bands via SPA...")
            
            # Parse Exclude Bands
            exclude_indices = []
            if self.exclude_bands_str:
                try:
                    for part in self.exclude_bands_str.split(','):
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            # UI is usually 1-based, Python is 0-based.
                            # User input: 1-40 -> Indices 0..39
                            exclude_indices.extend(range(start - 1, end))
                        else:
                            exclude_indices.append(int(part) - 1)
                except:
                    if not silent: self.log_message.emit("Warning: Failed to parse exclude bands string.")
            
            # SPA Downsampling (Optimization for Selection ONLY)
            # Use max 3000 samples for SPA speed, but train on FULL data
            # SPA on Full Dataset (User Request: No Downsampling)
            dummy_cube = X.reshape(-1, 1, X.shape[1])
                 
            # Pass exclude_indices to SPA
            selected_indices, scores, mean_spec = select_best_bands(
                dummy_cube, 
                n_bands=n_features, 
                method='spa',
                exclude_indices=exclude_indices
            )
            
            selected_bands_1based = [b + 1 for b in selected_indices]
            if not silent: 
                self.log_message.emit("-" * 40)
                self.log_message.emit(f"✅ Selected Spa Bands (Top {n_features}):")
                self.log_message.emit(f"   {selected_bands_1based}")
                self.log_message.emit("-" * 40)
            
            # 4. Train Model
            if not silent: self.log_message.emit(f"Training {model_type}...")
            
            X_sub = X[:, selected_indices]
            service = LearningService()
            model, acc = service.train_model(X_sub, y, model_type=model_type, test_ratio=test_ratio)
            
            if not silent: self.log_message.emit(f"Training Accuracy: {acc*100:.2f}%")
            self.progress_update.emit(100)
            
            # 5. Export
            # Construct colors map (Default to Green/Red/Blue...)
            colors_map = {str(k): "#00FF00" for k in label_map.keys()} 
            
            service.export_model(
                model, 
                selected_indices, 
                output_path, 
                preprocessing_config=self.prep_chain,
                processing_mode=self.processing_mode,
                mask_rules=self.mask_rules,
                label_map={str(k):v for k,v in label_map.items()},
                colors_map=colors_map,
                exclude_rules=self.exclude_bands_str, 
                threshold=self.threshold,
                mean_spectrum=mean_spec.tolist() if mean_spec is not None else None,
                spa_scores=scores.tolist() if scores is not None else None
            )
            
            self.log_message.emit(f"Model exported to {output_path}")
            self.training_finished.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"Worker Error: {e}")
            import traceback
            # self.log_message.emit(traceback.format_exc())
            self.training_finished.emit(False)

    def stop(self):
        self.is_running = False

    # --- Helper Private Methods ---

    def _validate_inputs(self):
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        valid_cnt = 0
        for name, files in self.file_groups.items():
            if len(files) > 0 and name.lower() not in EXCLUDED_NAMES:
                valid_cnt += 1
        
        if valid_cnt < 2:
            self.log_message.emit("Error: Need at least 2 valid classes for training!")
            self.training_finished.emit(False)
            return False
        return True

    def _log_configuration(self, silent, n_features):
        if silent: return
        self.log_message.emit("-" * 40)
        self.log_message.emit("[Preprocessing Configuration]")
        self.log_message.emit(f" • Mode: {self.processing_mode}")
        self.log_message.emit(f" • Threshold: {self.threshold}")
        self.log_message.emit(f" • Mask Rules: {self.mask_rules if self.mask_rules else 'None'}")
        
        if not self.prep_chain:
            self.log_message.emit(" • Chain: None (Raw)")
        else:
            for i, step in enumerate(self.prep_chain):
                name = step['name']
                p_str = ", ".join([f"{k}={v}" for k, v in step['params'].items()])
                self.log_message.emit(f" • Step {i+1}: {name} ({p_str})")
        self.log_message.emit("-" * 40)

    def _get_data_with_caching(self, silent):
        """
        Returns (X, y, label_map).
        Decides whether to use precomputed_data or load from files.
        """
        # Prepare Label Map first (needed in both cases)
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        valid_groups = {n: f for n, f in self.file_groups.items() if len(f) > 0 and n.lower() not in EXCLUDED_NAMES}
        valid_groups = dict(sorted(valid_groups.items()))  # Ensure deterministic order
        
        label_map = {}
        for label_id, (class_name, _) in enumerate(valid_groups.items()):
            label_map[label_id] = class_name

        # 1. Fast Path (Fully Precomputed)
        if self.precomputed_data is not None:
            if not silent: self.log_message.emit("⚡ Smart Cache Hit! Reusing processed data (Instant Start)...")
            X, y = self.precomputed_data
            return X, y, label_map
            
        # 2. Medium Path (Base Data Cache - Masked but need Prep)
        if self.base_data_cache is not None:
             if not silent: self.log_message.emit("⚡ Base Data Cache Hit! Reusing masked data, applying preprocessing...")
             X_base, y_base = self.base_data_cache
             
             X = ProcessingService.apply_preprocessing_chain(X_base, self.prep_chain)
             return X, y_base, label_map

        # 3. Slow Path
        if not silent: self.log_message.emit("Starting File Loading & Processing...")
        X, y = self._load_and_process_files(valid_groups, silent)
        return X, y, label_map

    def _load_and_process_files(self, valid_groups, silent):
        X_all = []
        y_all = []
        
        total_files = sum(len(f) for f in valid_groups.values())
        processed_cnt = 0
        
        for label_id, (class_name, files) in enumerate(valid_groups.items()):
            if not silent: self.log_message.emit(f"Processing Class '{class_name}' (ID={label_id})...")
            
            for f in files:
                if not self.is_running: return None, None
                
                try:
                    # Thread-Safe File Cache Check
                    if f in self.data_cache:
                        cube, waves = self.data_cache[f]
                    else:
                        cube, waves = load_hsi_data(f)
                        cube = np.nan_to_num(cube)
                        self.data_cache[f] = (cube, waves)
                    
                    cube = np.nan_to_num(cube)
                    
                    # 1. Get Base Data (Reflectance + Masked)
                    data_base, mask = ProcessingService.get_base_data(
                        cube,
                        mode=self.processing_mode,
                        threshold=self.threshold,
                        mask_rules=self.mask_rules,
                        white_ref=self.white_ref,
                        dark_ref=self.dark_ref
                    )
                    
                    if data_base.shape[0] > 0:
                        # Append to Base List
                        X_all.append(data_base)
                        y_all.append(np.full(data_base.shape[0], label_id))
                        # processed_cnt logic can remain same
                        
                except Exception as e:
                    if not silent: self.log_message.emit(f"Error processing {f}: {e}")
                
                processed_cnt += 1
                if not silent: self.progress_update.emit(int((processed_cnt / total_files) * 50))

        X_base = np.vstack(X_all)
        y_base = np.concatenate(y_all)
        
        # 3. Emit Base Cache (Handoff to Auto-ML or next run)
        if not silent: self.log_message.emit("💾 Caching Masked Data for future use...")
        self.data_ready.emit(X_base, y_base)
        
        # 4. Apply Preprocessing Chain
        try:
             # Need import here or at top? It's imported at top? Yes, in imports.
             # Wait, TrainingWorker imports ProcessingService.
             pass
        except: pass
        
        X_final = ProcessingService.apply_preprocessing_chain(X_base, self.prep_chain)
        
        if not silent: self.log_message.emit(f"Total Loaded Samples: {X_final.shape[0]}")
        return X_final, y_base
