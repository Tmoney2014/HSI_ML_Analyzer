import sys
import os
import json
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, QAction, QFileDialog, QMessageBox)
from PyQt5.QtCore import QTimer
from viewmodels.main_vm import MainViewModel
from viewmodels.analysis_vm import AnalysisViewModel
from viewmodels.training_vm import TrainingViewModel

from views.tabs.tab_data import TabData
from views.tabs.tab_analysis import TabAnalysis
from views.tabs.tab_training import TabTraining
from views.dialogs.project_welcome_dialog import ProjectWelcomeDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Professional Analyzer v3.1 (Project Based)")
        self.setGeometry(50, 50, 1600, 1000)
        
        # 1. Initialize View Models
        self.main_vm = MainViewModel()
        self.analysis_vm = AnalysisViewModel(self.main_vm)
        self.training_vm = TrainingViewModel(self.main_vm, self.analysis_vm)
        
        # Connect Signals for Auto-Save
        self.main_vm.files_changed.connect(self.auto_save_slot)
        self.main_vm.refs_changed.connect(self.auto_save_slot)
        self.analysis_vm.params_changed.connect(self.auto_save_slot) # Auto-save on Analysis Params
        self.main_vm.save_requested.connect(self.save_project) # Manual Save Button
        
        # 2. UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.create_menu_bar()
        
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
        self.status_bar.showMessage("Ready. Open or Create a Project.")
        
        # 4. Trigger Startup Dialog
        QTimer.singleShot(100, self.show_startup_dialog)

    def show_startup_dialog(self):
        """Show modal dialog to force Project Choice"""
        dlg = ProjectWelcomeDialog(self)
        if dlg.exec_():
            if dlg.choice == 'new':
                self.new_project_silent() # Don't ask for confirmation on fresh start
            elif dlg.choice == 'open':
                self.open_project_startup()
        else:
            # User closed dialog without choice -> Exit App
            print("User cancelled startup. Exiting.")
            self.close()

    def new_project_silent(self):
        """Start new project from dialog (Force Save First)"""
        f, _ = QFileDialog.getSaveFileName(self, "New Project - Save As", "", "HSI Project (*.json)")
        if not f:
            # User cancelled at startup.
            # Option: Show dialog again or exit. Let's just exit or return.
            # Return means app stays open in empty state? No, startup dialog logic handles return.
            # But here we are inside the main window method called by dialog logic.
            # If user cancels here, we haven't done anything.
            # Let's just return. The user can use Menu > New later.
            return

        self.main_vm.reset_session()
        self.main_vm.current_project_path = f
        self._do_save(f)
        
        self.setWindowTitle(f"HSI Professional Analyzer - {os.path.basename(f)}")
        self.status_bar.showMessage(f"New project created: {f}")

    def open_project_startup(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "HSI Project (*.json);;All Files (*)")
        if f:
            self.load_project_file(f)
        else:
            # User cancelled file selection -> Show dialog again or Exit?
            # Let's show dialog again to be nice.
            QTimer.singleShot(0, self.show_startup_dialog)

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('&File')
        
        new_act = QAction('New Project', self)
        new_act.setShortcut('Ctrl+N')
        new_act.triggered.connect(self.new_project)
        file_menu.addAction(new_act)
        
        open_act = QAction('Open Project...', self)
        open_act.setShortcut('Ctrl+O')
        open_act.triggered.connect(self.open_project)
        file_menu.addAction(open_act)
        
        save_act = QAction('Save Project', self)
        save_act.setShortcut('Ctrl+S')
        save_act.triggered.connect(self.save_project)
        file_menu.addAction(save_act)
        
        save_as_act = QAction('Save Project As...', self)
        save_as_act.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_act)
        
        file_menu.addSeparator()
        
        exit_act = QAction('Exit', self)
        exit_act.setShortcut('Ctrl+Q')
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    def new_project(self):
        reply = QMessageBox.question(self, 'New Project', 
                                     "Create a new project? Unsaved changes will be lost.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Enforce Save First Logic
            f, _ = QFileDialog.getSaveFileName(self, "New Project - Save As", "", "HSI Project (*.json)")
            if not f:
                return # User cancelled file creation
            
            self.main_vm.reset_session()
            
            # Set path and perform initial save
            self.main_vm.current_project_path = f
            self._do_save(f)
            
            self.setWindowTitle(f"HSI Professional Analyzer - {os.path.basename(f)}")
            self.status_bar.showMessage(f"New project created: {f}")

    def open_project(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "HSI Project (*.json);;All Files (*)")
        if not f: return
        self.load_project_file(f)

    def save_project(self):
        if self.main_vm.current_project_path:
            self._do_save(self.main_vm.current_project_path)
            self.status_bar.showMessage(f"Project saved: {self.main_vm.current_project_path}", 3000)
        else:
            self.save_project_as()

    def save_project_as(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "HSI Project (*.json)")
        if not f: return
        self._do_save(f)
        self.main_vm.current_project_path = f
        self.setWindowTitle(f"HSI Professional Analyzer - {os.path.basename(f)}")
        self.status_bar.showMessage(f"Project saved to {f}")

    def auto_save_slot(self):
        """Called automatically when data changes."""
        if self.main_vm.current_project_path:
            self._do_save(self.main_vm.current_project_path)
            # self.status_bar.showMessage("Auto-saved.", 1000) # Optional feedback

    def _do_save(self, path):
        # Extract SG params
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
            "file_groups": self.main_vm.file_groups,
            "group_colors": self.main_vm.group_colors,
            "white_ref": self.main_vm.white_ref,
            "dark_ref": self.main_vm.dark_ref,
            "use_ref": self.main_vm.use_ref,
            "threshold": str(self.analysis_vm.threshold),
            "mask_band": self.analysis_vm.mask_rules if self.analysis_vm.mask_rules else "Mean",
            "prep_chain": self.analysis_vm.prep_chain,
            "sg_win": sg_win,
            "sg_poly": sg_poly,
            "sg_deriv": sg_deriv
        }
        try:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            print(f"Save Failed: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{e}")

    def load_project_file(self, path):
        if not os.path.exists(path): return
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            
            # Logic similar to load_session but uses cfg
            if "file_groups" in cfg:
                self.main_vm.file_groups = cfg["file_groups"]
                self.main_vm.group_colors = cfg.get("group_colors", {})
                self.main_vm.files_changed.emit()
            elif "normal_files" in cfg:
                # Legacy compatibility import
                 self.main_vm.file_groups = {}
                 self.main_vm.add_group("Normal")
                 self.main_vm.add_files_to_group("Normal", cfg["normal_files"])
                 if "defect_files" in cfg:
                     self.main_vm.add_group("Defect")
                     self.main_vm.add_files_to_group("Defect", cfg["defect_files"])
                 self.main_vm.files_changed.emit()

            w_ref = cfg.get("white_ref", "")
            d_ref = cfg.get("dark_ref", "")
            if w_ref: self.main_vm.set_white_ref(w_ref)
            if d_ref: self.main_vm.set_dark_ref(d_ref)
            
            if "use_ref" in cfg: 
                self.main_vm.set_use_ref(cfg["use_ref"])
                if hasattr(self.tab_data, 'radio_ref'):
                     self.tab_data.radio_ref.setChecked(cfg["use_ref"])
                     self.tab_data.radio_raw.setChecked(not cfg["use_ref"])
            
            # Analysis Params
            if "threshold" in cfg: self.analysis_vm.threshold = float(cfg["threshold"])
            mask_val = cfg.get("mask_band") or cfg.get("mask_rules")
            if mask_val and mask_val != "Mean":
                self.analysis_vm.mask_rules = mask_val
            else:
                self.analysis_vm.mask_rules = None
            
            if "prep_chain" in cfg:
                self.analysis_vm.prep_chain = cfg["prep_chain"]
            
            # Restore UI
            self.tab_analysis.restore_ui()
            
            self.main_vm.current_project_path = path
            self.setWindowTitle(f"HSI Professional Analyzer - {os.path.basename(path)}")
            self.status_bar.showMessage(f"Project loaded: {path}")
            
            self.main_vm.files_changed.emit() # Force refresh
            
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load project:\n{e}")

    def closeEvent(self, event):
        # Auto-saved already. Just check if we have unsaved work?
        # If auto-save is on and path is set, we are safe.
        # If no path set and data exists, ask to save.
        if not self.main_vm.current_project_path and (self.main_vm.file_groups):
             reply = QMessageBox.question(self, 'Save Confirmation', 
                                     "Save changes to a project file before exit?",
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
             if reply == QMessageBox.Yes:
                 self.save_project_as()
                 if not self.main_vm.current_project_path: # Cancelled save
                     event.ignore()
                     return
             elif reply == QMessageBox.Cancel:
                 event.ignore()
                 return
        
        self.tab_analysis.close_all_windows()
        event.accept()

    # Removed legacy save_session methods
