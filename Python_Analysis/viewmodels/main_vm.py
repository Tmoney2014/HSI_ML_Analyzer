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
        # Dynamic Grouping: Key=GroupName, Value=List[str]
        self.file_groups = {} 
        # Group Colors: Key=GroupName, Value=HexColorString or QColor
        self.group_colors = {}
        
        self.white_ref: str = ""
        self.dark_ref: str = ""
        self.use_ref: bool = False
        
        # Cache for loaded cubes to avoid re-reading disk (path -> (cube, waves))
        self.data_cache = {}
        
        # Reference Data Cache (Loaded Arrays)
        self.cache_white: Optional[np.ndarray] = None
        self.cache_dark: Optional[np.ndarray] = None
        
        # Project Management
        self.current_project_path: Optional[str] = None

    def reset_session(self):
        """Reset all data for New Project"""
        self.file_groups.clear()
        self.group_colors.clear()
        self.white_ref = ""
        self.dark_ref = ""
        self.use_ref = False
        self.data_cache.clear()
        self.cache_white = None
        self.cache_dark = None
        self.current_project_path = None
        
        # Create Default 'Ignore' Group
        self.add_group("Ignore")
        self.set_group_color("Ignore", "#808080") # Gray color for ignored group
        
        self.files_changed.emit()
        self.refs_changed.emit()
        self.mode_changed.emit(False)

    def set_use_ref(self, enabled: bool):
        if self.use_ref != enabled:
            self.use_ref = enabled
            self.mode_changed.emit(enabled)

    # --- Group Management ---
    def add_group(self, name: str):
        if name not in self.file_groups:
            self.file_groups[name] = []
            # Assign random unique color
            import random
            rand_color = "#%06x" % random.randint(0, 0xFFFFFF)
            self.group_colors[name] = rand_color
            self.files_changed.emit()

    def remove_group(self, name: str):
        if name in self.file_groups:
            del self.file_groups[name]
            if name in self.group_colors:
                del self.group_colors[name]
            self.files_changed.emit()
            
    def rename_group(self, old_name: str, new_name: str):
        if old_name in self.file_groups and new_name not in self.file_groups:
            self.file_groups[new_name] = self.file_groups.pop(old_name)
            if old_name in self.group_colors:
                self.group_colors[new_name] = self.group_colors.pop(old_name)
            self.files_changed.emit()
            
    def set_group_color(self, name: str, color: str):
        if name in self.file_groups:
            self.group_colors[name] = color
            self.files_changed.emit()

    def add_files_to_group(self, group_name: str, paths: List[str]):
        if group_name in self.file_groups:
            # Avoid duplicates
            current_set = set(self.file_groups[group_name])
            for p in paths:
                if p not in current_set:
                    self.file_groups[group_name].append(p)
            self.files_changed.emit()

    def remove_files_from_group(self, group_name: str, paths: List[str]):
        if group_name in self.file_groups:
            self.file_groups[group_name] = [f for f in self.file_groups[group_name] if f not in paths]
            self.files_changed.emit()

    def get_all_files(self):
        all_files = []
        for files in self.file_groups.values():
            all_files.extend(files)
        return all_files

    def request_save(self):
        self.save_requested.emit()

    def move_file_to_group(self, file_path: str, target_group: str):
        """Move a file from its current group to a target group"""
        if target_group not in self.file_groups:
            return

        # Find current group
        source_group = None
        for name, files in self.file_groups.items():
            if file_path in files:
                source_group = name
                break
        
        if source_group == target_group:
            return # No change
            
        if source_group:
            self.file_groups[source_group].remove(file_path)
            
        # Add to target (avoid duplicates)
        if file_path not in self.file_groups[target_group]:
            self.file_groups[target_group].append(file_path)
            
        self.files_changed.emit()
