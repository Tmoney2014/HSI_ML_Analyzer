from PyQt5.QtCore import QObject, pyqtSignal
from typing import List, Optional
import numpy as np
import os
from services.data_loader import load_hsi_data
from collections import OrderedDict
import threading
try:
    import psutil
except ImportError:
    psutil = None

class SmartCache(OrderedDict):
    def __init__(self, max_items=10, min_memory_gb=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_items = max_items
        self.min_memory_gb = min_memory_gb
        self.lock = threading.RLock()
        
    def __getitem__(self, key):
        with self.lock:
            value = super().__getitem__(key)
            # Avoid re-ordering if we are in the middle of checking/evicting
            if not getattr(self, '_evicting', False):
                self.move_to_end(key) # Mark as recently used
            return value
        
    def __setitem__(self, key, value):
        with self.lock:
            if key in self and not getattr(self, '_evicting', False):
                self.move_to_end(key)
            super().__setitem__(key, value)
            self._check_limits()
    
    def __contains__(self, key):
        with self.lock:
            return super().__contains__(key)

    def get(self, key, default=None):
        with self.lock:
            return super().get(key, default)
        
    def _check_limits(self):
        # Prevent recursion or interference during eviction
        if getattr(self, '_evicting', False):
            return
            
        self._evicting = True
        try:
            # 1. Count Limit
            while len(self) > self.max_items:
                # print(f"[Cache] Count Limit Reached ({len(self)} > {self.max_items}). Evicting...")
                self.popitem(last=False) # Remove oldest (FIFO)
                
            # 2. Memory Limit (Safety Guard)
            if psutil:
                try:
                    mem = psutil.virtual_memory()
                    available_gb = mem.available / (1024 ** 3)
                    # If memory is critical, evict aggressively until safe or empty
                    while available_gb < self.min_memory_gb and len(self) > 0:
                        # print(f"[Cache] Low Memory ({available_gb:.2f}GB < {self.min_memory_gb}GB). Evicting...")
                        self.popitem(last=False)
                        mem = psutil.virtual_memory()
                        available_gb = mem.available / (1024 ** 3)
                except Exception as e:
                    print(f"[Cache] Memory check failed: {e}")
        finally:
            self._evicting = False
                
    def set_config(self, max_items, min_memory_gb):
        with self.lock:
            self.max_items = max_items
            self.min_memory_gb = min_memory_gb
            self._check_limits()

class MainViewModel(QObject):
    # Signals to notify Views of changes
    files_changed = pyqtSignal()
    refs_changed = pyqtSignal()
    mode_changed = pyqtSignal(str) # "Raw", "Reflectance", "Absorbance"
    save_requested = pyqtSignal() # New signal for manual save

    def __init__(self):
        super().__init__()
        # Dynamic Grouping: Key=GroupName, Value=List[str]
        self.file_groups = {} 
        # Group Colors: Key=GroupName, Value=HexColorString or QColor
        self.group_colors = {}
        
        self.white_ref: str = ""
        self.dark_ref: str = ""
        self.processing_mode: str = "Raw" # Default
        
        # Cache for loaded cubes to avoid re-reading disk (path -> (cube, waves))
        # Use SmartCache for LRU and Memory Protection
        self.data_cache = SmartCache(max_items=20, min_memory_gb=1.0)
        
        # Reference Data Cache (Loaded Arrays)
        self.cache_white: Optional[np.ndarray] = None
        self.cache_dark: Optional[np.ndarray] = None
        
        # Project Management
        self.current_project_path: Optional[str] = None
        
    def set_cache_config(self, limit: int, min_mem_gb: float):
        """Configure Smart Cache Limits"""
        if isinstance(self.data_cache, SmartCache):
            self.data_cache.set_config(limit, min_mem_gb)

    def reset_session(self):
        """Reset all data for New Project"""
        self.file_groups.clear()
        self.group_colors.clear()
        self.white_ref = ""
        self.dark_ref = ""
        self.processing_mode = "Raw"
        self.data_cache.clear()
        self.cache_white = None
        self.cache_dark = None
        self.current_project_path = None
        
        # Create Default 'Ignore' Group
        self.add_group("Ignore")
        self.set_group_color("Ignore", "#808080") # Gray color for ignored group
        
        self.files_changed.emit()
        self.refs_changed.emit()
        self.mode_changed.emit("Raw")

    def set_use_ref(self, enabled: bool):
        # Compatibility Shim
        mode = "Reflectance" if enabled else "Raw"
        self.set_processing_mode(mode)

    @property
    def use_ref(self) -> bool:
        return self.processing_mode in ["Reflectance", "Absorbance"]

    def set_processing_mode(self, mode: str):
        if mode not in ["Raw", "Reflectance", "Absorbance"]:
            return
            
        if self.processing_mode != mode:
            self.processing_mode = mode
            self.mode_changed.emit(mode)

    def set_white_ref(self, path: str):
        self.white_ref = path
        self.cache_white = None  # Invalidate cache
        self.refs_changed.emit()

    def set_dark_ref(self, path: str):
        self.dark_ref = path
        self.cache_dark = None # Invalidate cache
        self.refs_changed.emit()

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
