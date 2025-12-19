from PyQt5.QtCore import QObject, pyqtSignal
from typing import List, Optional
import numpy as np
import os
from services.data_loader import load_hsi_data

class MainViewModel(QObject):
    # Signals to notify Views of changes
    files_changed = pyqtSignal()
    refs_changed = pyqtSignal()
    mode_changed = pyqtSignal(bool) # True=Reflectance, False=Raw
    save_requested = pyqtSignal() # New signal for manual save

    def __init__(self):
        super().__init__()
        self.normal_files: List[str] = []
        self.defect_files: List[str] = []
        
        self.white_ref: str = ""
        self.dark_ref: str = ""
        self.use_ref: bool = False
        
        # Cache for loaded cubes to avoid re-reading disk (path -> (cube, waves))
        self.data_cache = {}
        
        # Reference Data Cache (Loaded Arrays)
        self.cache_white: Optional[np.ndarray] = None
        self.cache_dark: Optional[np.ndarray] = None

    def set_use_ref(self, enabled: bool):
        if self.use_ref != enabled:
            self.use_ref = enabled
            self.mode_changed.emit(enabled)

    def add_normal_files(self, paths: List[str]):
        self.normal_files.extend(paths)
        self.files_changed.emit()

    def add_defect_files(self, paths: List[str]):
        self.defect_files.extend(paths)
        self.files_changed.emit()
        
    def clear_normal_files(self):
        self.normal_files.clear()
        self.files_changed.emit()

    def clear_defect_files(self):
        self.defect_files.clear()
        self.files_changed.emit()

    def set_white_ref(self, path: str):
        self.white_ref = path
        # Immediate Load & Cache
        if path and os.path.exists(path):
            try:
                cube, _ = load_hsi_data(path)
                # Helper: Mean spectrum if it's a cube, or keep as is? 
                # Original app code: self.cache_white = np.mean(w, axis=0)
                self.cache_white = np.mean(cube, axis=0) # Shape (Bands,)
            except Exception as e:
                print(f"Error loading white ref: {e}")
                self.cache_white = None
        else:
            self.cache_white = None
            
        self.refs_changed.emit()

    def set_dark_ref(self, path: str):
        self.dark_ref = path
        if path and os.path.exists(path):
            try:
                cube, _ = load_hsi_data(path)
                # Original app code: self.cache_dark = np.mean(d, axis=0)
                self.cache_dark = np.mean(cube, axis=0)
            except Exception as e:
                print(f"Error loading dark ref: {e}")
                self.cache_dark = None
        else:
            self.cache_dark = None
            
        self.refs_changed.emit()
        
    def get_all_files(self):
        return self.normal_files + self.defect_files

    def request_save(self):
        self.save_requested.emit()
