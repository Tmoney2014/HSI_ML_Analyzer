import sys
import os
import json
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from viewmodels.main_vm import MainViewModel
from viewmodels.analysis_vm import AnalysisViewModel
from viewmodels.training_vm import TrainingViewModel

from views.tabs.tab_data import TabData
from views.tabs.tab_analysis import TabAnalysis
from views.tabs.tab_training import TabTraining

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Professional Analyzer v3.0 (MVVM)")
        self.setGeometry(50, 50, 1600, 1000)
        
        # 1. Initialize View Models
        self.main_vm = MainViewModel()
        self.analysis_vm = AnalysisViewModel(self.main_vm)
        self.training_vm = TrainingViewModel(self.main_vm, self.analysis_vm)
        
        # Connect Save Signal
        self.main_vm.save_requested.connect(self.save_session_manual)
        
        # 2. UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { height: 40px; width: 200px; font-size: 14px; font-weight: bold; }")
        layout.addWidget(self.tabs)
        
        # 3. Create Views
        self.tab_data = TabData(self.main_vm)
        self.tab_analysis = TabAnalysis(self.main_vm, self.analysis_vm)
        self.tab_training = TabTraining(self.training_vm)
        
        self.tabs.addTab(self.tab_data, "📂 Data Management")
        self.tabs.addTab(self.tab_analysis, "📊 Analysis & Visualization")
        self.tabs.addTab(self.tab_training, "🎓 Training")
        
        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Restore Session
        self.load_session()

    def closeEvent(self, event):
        # Save Session
        self.save_session()
        
        # Close child windows in analysis tab
        self.tab_analysis.close_all_windows()
        event.accept()

    def save_session_manual(self):
        if self.save_session():
             from PyQt5.QtWidgets import QMessageBox
             QMessageBox.information(self, "Save Settings", "Session settings saved successfully.\n(File list, Params, Preprocessing)")

    def save_session(self):
        # Extract SG params from chain if present
        sg_win = "5"
        sg_poly = "2"
        sg_deriv = "0"
        
        for step in self.analysis_vm.prep_chain:
            if step['name'] == 'SG':
                p = step.get('params', {})
                sg_win = str(p.get('win', 5))
                sg_poly = str(p.get('poly', 2))
                sg_deriv = str(p.get('deriv', 0))
                break

        cfg = {
            "normal_files": self.main_vm.normal_files,
            "defect_files": self.main_vm.defect_files,
            "white_ref": self.main_vm.white_ref,
            "dark_ref": self.main_vm.dark_ref,
            "use_ref": self.main_vm.use_ref,
            "threshold": str(self.analysis_vm.threshold),
            "mask_band": self.analysis_vm.mask_rules if self.analysis_vm.mask_rules else "Mean", # User requested 'mask_band'
            "prep_chain": self.analysis_vm.prep_chain,
            # Flat params for compatibility
            "sg_win": sg_win,
            "sg_poly": sg_poly,
            "sg_deriv": sg_deriv
        }
        try:
            with open("session_config.json", "w") as f:
                json.dump(cfg, f, indent=4)
            print("Session saved.")
            return True
        except Exception as e:
            print(f"Failed to save session: {e}")
            return False

    def load_session(self):
        if not os.path.exists("session_config.json"): return
        try:
            with open("session_config.json", "r") as f:
                cfg = json.load(f)
            
            # 1. Load Main VM Data
            if "normal_files" in cfg: self.main_vm.add_normal_files(cfg["normal_files"])
            if "defect_files" in cfg: self.main_vm.add_defect_files(cfg["defect_files"])
            
            w_ref = cfg.get("white_ref", "")
            d_ref = cfg.get("dark_ref", "")
            if w_ref: self.main_vm.set_white_ref(w_ref)
            if d_ref: self.main_vm.set_dark_ref(d_ref)
            
            # 2. Restore Mode
            if "use_ref" in cfg: 
                self.main_vm.set_use_ref(cfg["use_ref"])
                if hasattr(self.tab_data, 'radio_ref'):
                     self.tab_data.radio_ref.setChecked(cfg["use_ref"])
                     self.tab_data.radio_raw.setChecked(not cfg["use_ref"])
            
            # 3. Load Analysis VM Params
            if "threshold" in cfg: self.analysis_vm.threshold = float(cfg["threshold"])
            
            # Handle 'mask_band' alias
            mask_val = cfg.get("mask_band") or cfg.get("mask_rules")
            if mask_val and mask_val != "Mean":
                self.analysis_vm.mask_rules = mask_val
            else:
                self.analysis_vm.mask_rules = None
            
            if "prep_chain" in cfg:
                self.analysis_vm.prep_chain = cfg["prep_chain"]
            
            # UI Restoration
            self.tab_analysis.txt_thresh.setText(str(self.analysis_vm.threshold))
            if self.analysis_vm.mask_rules:
                self.tab_analysis.txt_mask_band.setText(str(self.analysis_vm.mask_rules))
            
            # Restore SG params legacy way if needed (will be handled by restore_ui if in chain, 
            # but if chain is missing, we might need to reconstruct it? 
            # The restore_ui in TabAnalysis iterates the chain. 
            # If loaded chain is empty but flat params exist, we should probably add SG to chain?
            # For now, assuming prep_chain is source of truth if present.)
            
            # Full UI Restore including Prep Chain
            self.tab_analysis.restore_ui()
            
            print(f"Session loaded.")
            
        except Exception as e:
            print(f"Failed to load session: {e}")
