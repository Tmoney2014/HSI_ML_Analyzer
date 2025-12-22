from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from viewmodels.main_vm import MainViewModel
from viewmodels.analysis_vm import AnalysisViewModel
from services.learning_service import LearningService
from services.data_loader import load_hsi_data
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
        
    def run_training(self, output_path: str):
        file_groups = self.main_vm.file_groups
        
        # Validation: Need at least 2 groups with files
        # Filter out "Unassigned" or "-" classes
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        
        valid_groups = {}
        for name, files in file_groups.items():
            if len(files) > 0:
                if name.lower() in EXCLUDED_NAMES:
                    self.log_message.emit(f"Skipping excluded class: '{name}'")
                    continue
                valid_groups[name] = files
        
        if len(valid_groups) < 2:
            self.log_message.emit("Error: Need at least 2 valid classes for training!")
            self.training_finished.emit(False)
            return

        total_files = sum(len(files) for files in valid_groups.values())
        self.log_message.emit(f"Starting Training... Classes: {len(valid_groups)}, Total Files: {total_files}")
        self.progress_update.emit(0)
        
        try:
            # Trigger Auto-Save before training
            self.main_vm.request_save()
            
            # 1. Prepare Data
            X_all = []
            y_all = []
            cnt = 0
            label_map = {} # ID -> Name
            
            # Helper to process file
            def process_files(files, label):
                nonlocal cnt
                for f in files:
                    try:
                        cube, waves = load_hsi_data(f)
                        cube = np.nan_to_num(cube)
                        
                        if self.analysis_vm.use_ref:
                             cube = self.analysis_vm._convert_to_ref(cube)
                             
                        mask = processing.create_background_mask(cube, self.analysis_vm.threshold, self.analysis_vm.mask_rules)
                        # Ensure mask is boolean
                        if mask.dtype != bool: mask = mask.astype(bool)
                        
                        data = processing.apply_mask(cube, mask)
                        
                        # Apply Chain
                        for step in self.analysis_vm.prep_chain:
                            name = step.get('name')
                            p = step.get('params', {})
                            if name == "SG": data = processing.apply_savgol(data, p.get('win'), p.get('poly'), p.get('deriv', 0))
                            elif name == "SimpleDeriv": data = processing.apply_simple_derivative(data, gap=p.get('gap', 5), order=p.get('order', 1))
                            elif name == "SNV": data = processing.apply_snv(data)
                            elif name == "L2": data = processing.apply_l2_norm(data)
                            elif name == "MinMax": data = processing.apply_minmax_norm(data)
                            elif name == "Center": data = processing.apply_mean_centering(data)
                        
                        data = np.nan_to_num(data)
                        
                        if data.shape[0] > 0:
                            # subsample if too large
                            if data.shape[0] > 1000:
                                idx = np.random.choice(data.shape[0], 1000, replace=False)
                                data = data[idx]
                                
                            X_all.append(data)
                            y_all.append(np.full(data.shape[0], label))
                    except Exception as e:
                        print(f"Error processing {f}: {e}")
                        
                    cnt += 1
                    self.progress_update.emit(int((cnt/total_files)*50))
                    
            # Auto-assign Labels
            colors_map = {}
            for idx, (name, files) in enumerate(valid_groups.items()):
                label_map[str(idx)] = name # JSON keys must be strings
                # Get color, default to white?
                colors_map[str(idx)] = self.main_vm.group_colors.get(name, "#FFFFFF")
                
                self.log_message.emit(f"Processing Class '{name}' (ID={idx})...")
                process_files(files, idx)
            
            if not X_all:
                self.log_message.emit("Error: No valid pixels found after masking!")
                self.training_finished.emit(False)
                return
                
            X_train = np.vstack(X_all)
            y_train = np.hstack(y_all)
            
            self.log_message.emit(f"Data Loaded. Samples: {X_train.shape[0]}")
            self.progress_update.emit(60)
            
            # 2. Band Selection (PCA)
            self.log_message.emit("Selecting best bands via PCA...")
            from utils.band_selection import select_best_bands
            
            dummy_cube = X_train[:min(5000, X_train.shape[0])].reshape(-1, 1, X_train.shape[1])
            selected_bands = select_best_bands(dummy_cube, n_bands=5)
            
            display_bands = [b + 1 for b in selected_bands]
            self.log_message.emit(f"Selected Bands (1-based): {display_bands}")
            
            X_train_sub = X_train[:, selected_bands]
            self.progress_update.emit(70)
            
            # 3. Train
            model, acc = self.service.train_svm(X_train_sub, y_train)
            self.log_message.emit(f"Training Complete. Accuracy: {acc*100:.2f}%")
            self.progress_update.emit(90)
            
            # 4. Export (Pass label_map and colors_map)
            self.service.export_model(
                model, 
                selected_bands, 
                output_path, 
                preprocessing_config=self.analysis_vm.prep_chain,
                use_ref=self.analysis_vm.use_ref,
                mask_rules=self.analysis_vm.mask_rules,
                label_map=label_map,
                colors_map=colors_map
            )
            self.log_message.emit(f"Saved to {output_path}")
            self.progress_update.emit(100)
            self.training_finished.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.training_finished.emit(False)
