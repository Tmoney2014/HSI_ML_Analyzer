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
        normal_files = self.main_vm.normal_files
        defect_files = self.main_vm.defect_files
        
        if not normal_files or not defect_files:
            self.log_message.emit("Error: Missing Training Data!")
            self.training_finished.emit(False)
            return

        self.log_message.emit(f"Starting Training... Normal: {len(normal_files)}, Defect: {len(defect_files)}")
        self.progress_update.emit(0)
        
        try:
            # 1. Prepare Data
            X_all = []
            y_all = []
            total = len(normal_files) + len(defect_files)
            cnt = 0
            
            # Helper to process file
            def process_files(files, label):
                nonlocal cnt
                for f in files:
                    # Reuse AnalysisVM logic for consistency or direct? 
                    # Direct is faster if we just check AnalysisVM's config
                    cube, waves = load_hsi_data(f)
                    
                    # Apply Preprocessing (Simulated from AnalysisVM config)
                    # Ideally, AnalysisVM should expose a 'process_cube(cube)' method that returns valid pixels
                    # For now, let's implement a simplified pipeline here matching AnalysisVM
                    
                    cube = np.nan_to_num(cube)
                    
                    if self.analysis_vm.use_ref:
                         cube = self.analysis_vm._convert_to_ref(cube)
                         
                    mask = processing.create_background_mask(cube, self.analysis_vm.threshold, self.analysis_vm.mask_rules)
                    data = processing.apply_mask(cube, mask)
                    
                    # Apply Chain
                    for step in self.analysis_vm.prep_chain:
                        name = step.get('name')
                        p = step.get('params', {})
                        if name == "SG": data = processing.apply_savgol(data, p.get('win'), p.get('poly'), p.get('deriv', 0))
                        elif name == "SimpleDeriv": data = processing.apply_simple_derivative(data, gap=p.get('gap', 5))
                        elif name == "SNV": data = processing.apply_snv(data)
                        elif name == "L2": data = processing.apply_l2_norm(data)
                        elif name == "MinMax": data = processing.apply_minmax_norm(data)
                        elif name == "Center": data = processing.apply_mean_centering(data)
                    
                    data = np.nan_to_num(data)
                    
                    if data.shape[0] > 0:
                        # subsample if too large?
                        if data.shape[0] > 1000:
                            idx = np.random.choice(data.shape[0], 1000, replace=False)
                            data = data[idx]
                            
                        X_all.append(data)
                        y_all.append(np.full(data.shape[0], label))
                        
                    cnt += 1
                    self.progress_update.emit(int((cnt/total)*50)) # First 50% is loading
                    
            process_files(normal_files, 0)
            process_files(defect_files, 1)
            
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
            
            # Reshape for band selection: (samples, 1, bands) -> (samples, 1, bands) is close enough
            # select_best_bands expects (H, W, Bands) but can accept (N, 1, B) as a cube
            dummy_cube = X_train[:min(5000, X_train.shape[0])].reshape(-1, 1, X_train.shape[1])
            selected_bands = select_best_bands(dummy_cube, n_bands=5)
            
            display_bands = [b + 1 for b in selected_bands]
            self.log_message.emit(f"Selected Bands (1-based): {display_bands}")
            
            # Subset to selected bands
            X_train_sub = X_train[:, selected_bands]
            self.progress_update.emit(70)
            
            # 3. Train
            model, acc = self.service.train_svm(X_train_sub, y_train)
            self.log_message.emit(f"Training Complete. Accuracy: {acc*100:.2f}%")
            self.progress_update.emit(90)
            
            # 4. Export
            self.service.export_model(
                model, 
                selected_bands, 
                output_path, 
                preprocessing_config=self.analysis_vm.prep_chain,
                use_ref=self.analysis_vm.use_ref,
                mask_rules=self.analysis_vm.mask_rules
            )
            self.log_message.emit(f"Saved to {output_path}")
            self.progress_update.emit(100)
            self.training_finished.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            self.training_finished.emit(False)
