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
    AI가 수정함: 캐시 구조 통합 - Base Data만 캐시
    """
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    training_finished = pyqtSignal(bool)
    base_data_ready = pyqtSignal(object, object)  # AI가 수정함: Base Data (X_base, y) 캐시용
    
    def __init__(self, file_groups, analysis_vm_state, main_vm_cache, params, colors_map=None):
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
        # AI가 수정함: precomputed_data 제거, base_data_cache만 사용
        self.base_data_cache = params.get('base_data_cache')  # (X_base, y) - NO Preprocessing
        self.is_running = True
        
        # AI가 수정함: 색상 맵 저장
        self.colors_map = colors_map

    def run(self):
        try:
            # 1. Setup & Validation
            if not self._validate_inputs(): return

            output_path = self.params['output_path']
            n_features = self.params['n_features']
            model_type = self.params['model_type']
            model_type = self.params['model_type']
            test_ratio = self.params['test_ratio']
            silent = self.params.get('silent', False)
            
            # AI가 추가함: Naming Metadata
            model_name = self.params.get('model_name', 'model')
            model_desc = self.params.get('model_desc', '')
            
            # Log Params
            self._log_configuration(silent, n_features)

            # 2. Data Acquisition (Smart Cache Check)
            self.progress_update.emit(0)
            X, y, label_map, base_data_emitted = self._get_data_with_caching(silent)
            
            if X is None or not self.is_running: 
                # AI가 수정함: Deadlock 방지 - 로드 실패 시 반드시 종료 신호 전송
                self.log_message.emit("Training Aborted or Data Load Failed.")
                self.training_finished.emit(False)
                return

            # AI가 수정함: Base Data 캐시 emit (전처리 전 데이터)
            if base_data_emitted:
                self.base_data_ready.emit(base_data_emitted[0], base_data_emitted[1])

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
            
            # AI가 수정함: Gap Diff로 밴드 수가 줄어들면 범위 초과 인덱스 무시
            n_bands = X.shape[1]
            exclude_indices = [i for i in exclude_indices if 0 <= i < n_bands]
            
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
            # AI가 수정함: 로그 콜백 연결 (UI 출력)
            # AI가 수정함: 로그 콜백 연결 (UI 출력) 및 Metrics 수신
            model, metrics = service.train_model(
                X_sub, y, 
                model_type=model_type, 
                test_ratio=test_ratio,
                log_callback=self.log_message.emit if not silent else None
            )
            
            acc = metrics['TestAccuracy']
            if not silent: self.log_message.emit(f"Training Accuracy: {acc}%")
            self.progress_update.emit(100)
            
            # 5. Export
            # Construct colors map
            if self.colors_map:
                # self.colors_map: {GroupName: Color}
                # label_map: {ClassID: GroupName}
                # Target: {ClassID: Color}
                final_colors_map = {}
                for cid, gname in label_map.items():
                    # gname might be int or str depending on logic, ensure match
                    if gname in self.colors_map:
                        final_colors_map[str(cid)] = self.colors_map[gname]
                    else:
                        final_colors_map[str(cid)] = "#00FF00" # Default
            else:
                final_colors_map = {str(k): "#00FF00" for k in label_map.keys()} 
            
            service.export_model(
                model, 
                selected_indices, 
                output_path, 
                preprocessing_config=self.prep_chain,
                processing_mode=self.processing_mode,
                mask_rules=self.mask_rules,
                label_map={str(k):v for k,v in label_map.items()},
                colors_map=final_colors_map,
                exclude_rules=self.exclude_bands_str, 
                threshold=self.threshold,
                mean_spectrum=mean_spec.tolist() if mean_spec is not None else None,
                spa_scores=scores.tolist() if scores is not None else None,

                metrics=metrics, # AI가 수정함: 성적표 전달
                # AI가 추가함: Metadata
                model_name=model_name,
                description=model_desc
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
        AI가 수정함: 캐시 구조 통합
        Returns (X_preprocessed, y, label_map, base_data_to_emit).
        base_data_to_emit: (X_base, y) if new data was loaded, None if cache was used.
        """
        # Prepare Label Map first (needed in both cases)
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        valid_groups = {n: f for n, f in self.file_groups.items() if len(f) > 0 and n.lower() not in EXCLUDED_NAMES}
        valid_groups = dict(sorted(valid_groups.items()))  # Ensure deterministic order
        
        label_map = {}
        for label_id, (class_name, _) in enumerate(valid_groups.items()):
            label_map[label_id] = class_name

        # 1. Cache Hit Path (Base Data 캐시 있음 → 전처리만 적용)
        if self.base_data_cache is not None:
             if not silent: self.log_message.emit("⚡ Base Data Cache Hit! Reusing masked data, applying preprocessing...")
             X_base, y_base = self.base_data_cache
             
             X = ProcessingService.apply_preprocessing_chain(X_base, self.prep_chain)
             return X, y_base, label_map, None  # 캐시 사용 → emit 안함

        # 2. Cache Miss Path (파일 로드 → Base Data 생성 → 전처리 적용)
        if not silent: self.log_message.emit("Starting File Loading & Processing...")
        X_base, y = self._load_and_process_base_data(valid_groups, silent)
        
        if X_base is None:
            return None, None, label_map, None
        
        # 전처리 적용
        X = ProcessingService.apply_preprocessing_chain(X_base, self.prep_chain)
        
        return X, y, label_map, (X_base, y)  # 새 Base Data → emit

    def _load_and_process_base_data(self, valid_groups, silent):
        """
        AI가 수정함: Base Data만 반환 (전처리는 호출자에서 적용)
        Returns (X_base, y) - NO Preprocessing applied.
        """
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
                    
                    # Get Base Data (Reflectance + Masked, NO Preprocessing)
                    data_base, mask = ProcessingService.get_base_data(
                        cube,
                        mode=self.processing_mode,
                        threshold=self.threshold,
                        mask_rules=self.mask_rules,
                        white_ref=self.white_ref,
                        dark_ref=self.dark_ref
                    )
                    
                    if data_base.shape[0] > 0:
                        X_all.append(data_base)
                        y_all.append(np.full(data_base.shape[0], label_id))
                        
                except Exception as e:
                    if not silent: self.log_message.emit(f"Error processing {f}: {e}")
                
                processed_cnt += 1
                if not silent: self.progress_update.emit(int((processed_cnt / total_files) * 50))

        if not X_all:
            return None, None
            
        X_base = np.vstack(X_all)
        y_base = np.concatenate(y_all)
        
        if not silent: self.log_message.emit(f"Total Loaded Samples: {X_base.shape[0]}")
        return X_base, y_base
