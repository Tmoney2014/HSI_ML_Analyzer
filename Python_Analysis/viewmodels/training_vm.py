from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from viewmodels.main_vm import MainViewModel
from viewmodels.analysis_vm import AnalysisViewModel
from services.learning_service import LearningService
from services.data_loader import load_hsi_data
from services.optimization_service import OptimizationService
from models import processing
import os # Added fix for NameError

class TrainingViewModel(QObject):
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    training_finished = pyqtSignal(bool) # Success?

    def __init__(self, main_vm: MainViewModel, analysis_vm: AnalysisViewModel):
        super().__init__()
        self.main_vm = main_vm
        self.analysis_vm = analysis_vm # Need access to prep strategy
        self.main_vm = main_vm
        self.analysis_vm = analysis_vm # Need access to prep strategy
        self.service = LearningService()
        self.optimizer = OptimizationService()
        self.optimizer.log_message.connect(self.log_message.emit)
        
    def run_optimization(self, output_path: str, model_type: str = "Linear SVM", test_ratio: float = 0.2):
        """
        Orchestrate Auto-ML Optimization.
        """
        if self.analysis_vm is None: return
        
        # 1. Capture Current State as 'Initial Params'
        prep_chain_copy = []
        for step in self.analysis_vm.prep_chain:
            # Deep copy params
            new_step = {'name': step['name'], 'params': step['params'].copy()}
            prep_chain_copy.append(new_step)
            
        initial_params = {
            'prep': prep_chain_copy,
            'n_features': 5, # Default Start
            # Add other context if needed
        }
        
        self.log_message.emit(f"=== Starting Auto-Optimization ({model_type}) ===")
        self.progress_update.emit(0)
        
        # Define Callback: Params -> Accuracy
        def trial_callback(params):
            # 1. Temporarily apply params to AnalysisVM (Virtual)
            # Actually, run_training uses self.analysis_vm.prep_chain directly.
            # So we need to inject these params into the training process.
            # Best way: Pass 'params' to run_training explicitly or mock AnalysisVM.
            
            # Since refactoring run_training is risky, let's Temporarily OVERWRITE AnalysisVM state
            # and Restore it later. This is the simplest way given the architecture.
            
            original_chain = self.analysis_vm.prep_chain
            
            # Apply Trial Params
            self.analysis_vm.prep_chain = params['prep']
            n_feat = params.get('n_features', 5)
            
            # Run Training (Internal Mode - No File Export, Just Score)
            # WARNING: run_training runs async? No, it seems synchronous in current code except for signals.
            # But the current implementation emits signals and returns.
            # We need a synchronous version or modify run_training to return score.
            
            # Modifying run_training to return score if requested.
            score = self.run_training(output_path=None, n_features=n_feat, internal_sim=True, silent=True, model_type=model_type, test_ratio=test_ratio)
            
            # Restore State
            self.analysis_vm.prep_chain = original_chain
            return score

        # Run Optimizer (Blocking for now, or thread?)
        # For simplicity, running on main thread (might freeze UI, but OK for now as requested)
        # Ideally should be threaded.
        
        try:
            best_params, history = self.optimizer.run_optimization(initial_params, trial_callback)
            
            # Apply Best Params to UI
            self.analysis_vm.set_preprocessing_chain(best_params['prep'])
            # Notify UI to update? (AnalysisVM should handle this if property setter used)
            # If not, we might need to refresh UI.
            
            self.log_message.emit(f"=== Optimization Finished. Best Accuracy: {history[-1][1]:.2f}% ===")
            
            # --- FINAL PRODUCTION RUN ---
            self.log_message.emit("\n[Final Phase] Running production training with best parameters...")
            # Ensure best params are applied (already done by line 84)
            best_n_features = best_params.get('n_features', 5)
            
            # Run final training (Export Model + Save Plots)
            self.run_training(output_path, n_features=best_n_features, internal_sim=False, model_type=model_type, test_ratio=test_ratio)
            
            # Note: training_finished will be emitted by run_training, so we don't emit it here
            # self.training_finished.emit(True) 
            
        except Exception as e:
            self.log_message.emit(f"Optimization Error: {e}")
            self.training_finished.emit(False)
            
    def run_training(self, output_path: str, n_features: int = 5, internal_sim: bool = False, silent: bool = False, model_type: str = "Linear SVM", test_ratio: float = 0.2):
        file_groups = self.main_vm.file_groups
        
        # Validation: Need at least 2 groups with files
        # Filter out "Unassigned" or "-" classes
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        
        valid_groups = {}
        for name, files in file_groups.items():
            if len(files) > 0:
                if name.lower() in EXCLUDED_NAMES:
                    if not silent: self.log_message.emit(f"Skipping excluded class: '{name}'")
                    continue
                valid_groups[name] = files
        
        if len(valid_groups) < 2:
            self.log_message.emit("Error: Need at least 2 valid classes for training!")
            self.training_finished.emit(False)
            return

        total_files = sum(len(files) for files in valid_groups.values())
        if not silent: 
            self.log_message.emit(f"Starting Training... Classes: {len(valid_groups)}, Total Files: {total_files}, Top-K Features: {n_features}")
            self.log_message.emit(f"   Model: {model_type}, Test Ratio: {test_ratio}")
        
        # Log Preprocessing Params
        param_log = [f"Bands={n_features}"]
        for step in self.analysis_vm.prep_chain:
            name = step['name']
            p_str = ", ".join([f"{k}={v}" for k, v in step['params'].items()])
            param_log.append(f"{name}({p_str})")
        if not silent:
            self.log_message.emit(f"   [Params] {', '.join(param_log)}")
        
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
                            elif name == "SimpleDeriv": data = processing.apply_simple_derivative(data, gap=p.get('gap', 5), order=p.get('order', 1), apply_ratio=p.get('ratio', False), ndi_threshold=p.get('ndi_threshold', 1e-4))
                            elif name == "SNV": data = processing.apply_snv(data)
                            elif name == "3PointDepth": data = processing.apply_rolling_3point_depth(data, gap=p.get('gap', 5))
                            elif name == "L2": data = processing.apply_l2_norm(data)
                            elif name == "MinSub": data = processing.apply_min_subtraction(data)
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
                    if not silent: self.progress_update.emit(int((cnt/total_files)*50))
                    
            # Auto-assign Labels
            colors_map = {}
            for idx, (name, files) in enumerate(valid_groups.items()):
                label_map[str(idx)] = name # JSON keys must be strings
                # Get color, default to white?
                colors_map[str(idx)] = self.main_vm.group_colors.get(name, "#FFFFFF")
                
                if not silent: self.log_message.emit(f"Processing Class '{name}' (ID={idx})...")
                process_files(files, idx)
            
            if not X_all:
                self.log_message.emit("Error: No valid pixels found after masking!")
                self.training_finished.emit(False)
                return
                
            X_train = np.vstack(X_all)
            y_train = np.hstack(y_all)
            
            if not silent: self.log_message.emit(f"Data Loaded. Samples: {X_train.shape[0]}")
            self.progress_update.emit(60)
            
            # 2. Band Selection (SPA)
            exclude_indices = self.analysis_vm.parse_exclude_bands()
            if exclude_indices:
                if not silent: self.log_message.emit(f"Excluding {len(exclude_indices)} bands from selection...")
                
            if not silent: self.log_message.emit(f"Selecting best {n_features} bands via SPA (Successive Projections Algorithm)...")
            from utils.band_selection import select_best_bands
            import matplotlib.pyplot as plt

            dummy_cube = X_train[:min(3000, X_train.shape[0])].reshape(-1, 1, X_train.shape[1])
            selected_bands, scores, mean_spec = select_best_bands(dummy_cube, n_bands=n_features, method='spa', exclude_indices=exclude_indices)
            
            display_bands = [b + 1 for b in selected_bands]
            if not silent: self.log_message.emit(f"Selected Bands (1-based): {display_bands}")
            
            # --- Visualization (Explainable AI) ---
            try:
                plt.figure(figsize=(10, 6))
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                # Plot 1: Importance Scores (Bar)
                x_axis = np.arange(1, len(scores) + 1)
                
                # Define colors: default skyblue, selected red
                bar_colors = ['skyblue'] * len(scores)
                for b_idx in selected_bands:
                    if b_idx < len(bar_colors):
                        bar_colors[b_idx] = 'red'
                
                # Plot Bars
                bars = ax1.bar(x_axis, scores, color=bar_colors, alpha=0.8, zorder=3)
                ax1.set_xlabel('Band Index')
                ax1.set_ylabel('Selectivity Score (SPA)', color='blue')

                # 1. Shade Excluded Regions (Visualizing the Void)
                if exclude_indices:
                    # Logic to group consecutive indices for efficient shading
                    sorted_ex = sorted(list(exclude_indices))
                    if sorted_ex:
                        ranges = []
                        start = sorted_ex[0]
                        end = sorted_ex[0]
                        for i in range(1, len(sorted_ex)):
                            if sorted_ex[i] == end + 1:
                                end = sorted_ex[i]
                            else:
                                ranges.append((start, end))
                                start = sorted_ex[i]
                                end = sorted_ex[i]
                        ranges.append((start, end))
                        
                        for (s, e) in ranges:
                            # 0-based index s -> plot x-coordinate s+1
                            # Shading from s+0.5 to e+1.5 to cover the integer slots
                            ax1.axvspan(s + 0.5, e + 1.5, color='#CCCCCC', alpha=0.5, zorder=1)

                
                # Custom Legend for Bars & Shading
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', label='Selected')
                ]
                if exclude_indices:
                    legend_elements.append(Patch(facecolor='#CCCCCC', label='Excluded Region', alpha=0.5))
                    
                ax1.legend(handles=legend_elements, loc='upper left')

                # Plot 2: Mean Spectrum (Line)
                ax2.plot(x_axis, mean_spec, color='black', linestyle='--', alpha=0.5, label='Mean Spectrum')
                ax2.set_ylabel('Intensity', color='gray')
                # ax2 legend
                ax2.legend(loc='upper right')
                
                # Highlight Selected with Labels (No vertical lines)
                for b_idx in selected_bands:
                    # Add text label above the bar
                    if b_idx < len(scores):
                        height = scores[b_idx]
                        ax1.text(b_idx + 1, height, f"{b_idx+1}", ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                
                plt.title(f"Band Selection Result (SPA Algorithm, Top-{n_features})")
                
                plt.title(f"Band Selection Result (SPA Algorithm, Top-{n_features})")
                
                # Save Plot (Skip if internal simulation)
                if output_path:
                    plot_path = os.path.join(os.path.dirname(output_path), "band_importance.png")
                    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                    plt.savefig(plot_path)
                    plt.close()
                    self.log_message.emit(f"   Ref: Importance plot saved to '{os.path.basename(plot_path)}'")
            except Exception as e:
                pass # Silent fail or log if needed, but plotting is secondary
            
            X_train_sub = X_train[:, selected_bands]
            self.progress_update.emit(70)
            # 4. Train Model
            if not silent: self.log_message.emit(f"Training {model_type}...")
            model, acc = self.service.train_model(X_train_sub, y_train, model_type=model_type, test_ratio=test_ratio)
            
            if not silent: self.log_message.emit(f"Training Accuracy: {acc*100:.2f}%")
            self.progress_update.emit(100)
            
            if internal_sim:
                return acc * 100.0 # Return Score immediately
            
            # 5. Export Model & Configlabel_map and colors_map)
            self.service.export_model(
                model, 
                selected_bands, 
                output_path, 
                preprocessing_config=self.analysis_vm.prep_chain,
                use_ref=self.analysis_vm.use_ref,
                mask_rules=self.analysis_vm.mask_rules,
                label_map=label_map,
                colors_map=colors_map,
                exclude_rules=self.analysis_vm.exclude_bands_str
            )
            self.log_message.emit(f"Saved to {output_path}")
            self.progress_update.emit(100)
            self.training_finished.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.training_finished.emit(False)
