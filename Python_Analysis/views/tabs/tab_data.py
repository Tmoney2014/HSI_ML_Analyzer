import os
import random
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QListWidget, QPushButton, QLabel, QLineEdit, QFileDialog, 
                             QListWidgetItem, QInputDialog, QMessageBox, QSplitter, QMenu, QColorDialog)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap, QColor
from viewmodels.main_vm import MainViewModel

class ClassListWidget(QListWidget):
    def __init__(self, vm: MainViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly) # Only accept drops, don't drag classes out

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or isinstance(event.source(), QListWidget):
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        # Determine target group
        item = self.itemAt(event.pos())
        if not item:
            # If dropped on empty space, maybe ignore or default?
            # User wants "Unassigned"? No, let's ignore for now.
            event.ignore()
            return
            
        target_group = item.text().split(" (")[0] # Extract "Name" from "Name (count)"
        
        # Handle Internal Move (From File List)
        if isinstance(event.source(), QListWidget):
            # Get selected items from source
            files = []
            for i in event.source().selectedItems():
                f_path = i.data(Qt.UserRole)
                if f_path:
                    files.append(f_path)
            
            if files:
                # Move files
                for f in files:
                    self.vm.move_file_to_group(f, target_group)
                
                event.accept()
                return

        event.ignore()

class TabData(QWidget):
    def __init__(self, main_vm: MainViewModel):
        super().__init__()
        self.vm = main_vm
        self.init_ui()
        
        # Connect Signals
        self.vm.files_changed.connect(self._refresh_lists)
        self.vm.refs_changed.connect(self._refresh_refs)
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Splitter to resize panels
        splitter = QSplitter(Qt.Horizontal)
        
        # --- 1. Class (Group) Management ---
        grp_classes = QGroupBox("1. Class List (Groups)")
        vbox_c = QVBoxLayout()
        
        # Custom List Widget for Drop Handling
        self.list_groups = ClassListWidget(self.vm)
        self.list_groups.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_groups.customContextMenuRequested.connect(self.show_group_context_menu)
        self.list_groups.itemDoubleClicked.connect(self.rename_group_item)
        self.list_groups.itemSelectionChanged.connect(self.on_group_selected)
        vbox_c.addWidget(self.list_groups)
        
        vbox_c.addWidget(QLabel("💡 Tip: Double-click to rename, Right-click for colors."))
        vbox_c.addWidget(QLabel("💡 Drag files here to move them."))
        
        hbox_btn_c = QHBoxLayout()
        btn_add_g = QPushButton("Add Class (+)")
        btn_add_g.clicked.connect(self.add_group_click)
        btn_del_g = QPushButton("Remove Class (-)")
        btn_del_g.clicked.connect(self.remove_group_click)
        hbox_btn_c.addWidget(btn_add_g)
        hbox_btn_c.addWidget(btn_del_g)
        vbox_c.addLayout(hbox_btn_c)
        grp_classes.setLayout(vbox_c)
        splitter.addWidget(grp_classes)
        
        # --- 2. File Management (for selected group) ---
        grp_files = QGroupBox("2. Files in Selected Class")
        vbox_f = QVBoxLayout()
        
        self.lbl_selected_group = QLabel("No class selected")
        self.lbl_selected_group.setStyleSheet("font-weight: bold; color: blue;")
        vbox_f.addWidget(self.lbl_selected_group)
        
        self.list_files = QListWidget()
        self.list_files.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_files.setDragDropMode(QListWidget.DragOnly) # Enable dragging files out
        self.list_files.setDragEnabled(True) # Explicitly enable drag
        
        # ... (rest of init_ui)
        vbox_f.addWidget(self.list_files)
        
        hbox_btn_f = QHBoxLayout()
        self.btn_add_f = QPushButton("Add Files (+)")
        self.btn_add_f.clicked.connect(self.add_files_click)
        self.btn_add_f.setEnabled(False)
        
        self.btn_del_f = QPushButton("Remove Files (-)")
        self.btn_del_f.clicked.connect(self.remove_files_click)
        self.btn_del_f.setEnabled(False)
        
        hbox_btn_f.addWidget(self.btn_add_f)
        hbox_btn_f.addWidget(self.btn_del_f)
        vbox_f.addLayout(hbox_btn_f)
        grp_files.setLayout(vbox_f)
        
        # Splitter for Resizable Layout (Groups and Files)
        splitter_left = QSplitter(Qt.Horizontal)
        splitter_left.addWidget(grp_classes)
        splitter_left.addWidget(grp_files)
        
        # KEY FIX: Prevent collapsing (disappearing)
        splitter_left.setCollapsible(0, False)
        splitter_left.setCollapsible(1, False)
        
        # Set Minimum Widths to prevent shrinking too much
        grp_classes.setMinimumWidth(250)
        grp_files.setMinimumWidth(400)
        
        # Set Ratio 1:2
        splitter_left.setStretchFactor(0, 1)
        splitter_left.setStretchFactor(1, 2)
        
        # Add Splitter to Main Layout
        layout.addWidget(splitter_left, stretch=4) # 4/5 width for the left two panels
        splitter_left.setStretchFactor(1, 2)
        
        # Add Splitter to Main Layout
        layout.addWidget(splitter_left, stretch=4) # 4/5 width for the left two panels
        
        # --- 3. References (Fixed Width) ---
        grp_ref = QGroupBox("3. Reference Setup")
        grp_ref.setFixedWidth(350)
        vbox_r = QVBoxLayout()
        
        vbox_r.addWidget(QLabel("White Reference (.hdr):"))
        self.txt_white = QLineEdit()
        hbox_w = QHBoxLayout()
        hbox_w.addWidget(self.txt_white)
        btn_w = QPushButton("...")
        btn_w.clicked.connect(self.browse_white)
        hbox_w.addWidget(btn_w)
        vbox_r.addLayout(hbox_w)
        
        vbox_r.addWidget(QLabel("Dark Reference (.hdr):"))
        self.txt_dark = QLineEdit()
        hbox_d = QHBoxLayout()
        hbox_d.addWidget(self.txt_dark)
        btn_d = QPushButton("...")
        btn_d.clicked.connect(self.browse_dark)
        hbox_d.addWidget(btn_d)
        vbox_r.addLayout(hbox_d)
        
        vbox_r.addSpacing(20)
        
        # Mode Toggle
        grp_mode = QGroupBox("Processing Mode")
        vbox_m = QVBoxLayout()
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        self.radio_raw = QRadioButton("Raw Data (DN)")
        self.radio_ref = QRadioButton("Reflectance")
        self.radio_abs = QRadioButton("Pseudo-Absorbance (-log)")
        self.radio_raw.setChecked(True)
        
        self.bg = QButtonGroup(self)
        self.bg.addButton(self.radio_raw)
        self.bg.addButton(self.radio_ref)
        self.bg.addButton(self.radio_abs)
        
        self.bg.buttonClicked.connect(self.on_mode_change)
        
        vbox_m.addWidget(self.radio_raw)
        vbox_m.addWidget(self.radio_ref)
        vbox_m.addWidget(self.radio_abs)
        grp_mode.setLayout(vbox_m)
        vbox_r.addWidget(grp_mode)
        
        vbox_r.addStretch()
        
        # Save Session Button
        btn_save = QPushButton("Save Settings (Session)")
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 30px;")
        btn_save.clicked.connect(lambda: self.vm.request_save())
        vbox_r.addWidget(btn_save)
        
        grp_ref.setLayout(vbox_r)
        layout.addWidget(grp_ref)
        
    def on_mode_change(self, btn):
        if self.radio_ref.isChecked() or self.radio_abs.isChecked():
            if not self.vm.white_ref or not os.path.exists(self.vm.white_ref):
                QMessageBox.warning(self, "Error", "Cannot switch mode!\nPlease load a White Reference first.")
                self.radio_raw.setChecked(True)
                self.vm.set_processing_mode("Raw")
                return
            
            if self.radio_ref.isChecked():
                self.vm.set_processing_mode("Reflectance")
            else:
                self.vm.set_processing_mode("Absorbance")
        else:
            self.vm.set_processing_mode("Raw")

    # --- Group Handlers ---
    def add_group_click(self):
        text, ok = QInputDialog.getText(self, "New Class", "Enter class name:")
        if ok and text:
            if text in self.vm.file_groups:
                QMessageBox.warning(self, "Duplicate", "Class name already exists.")
                return
            self.vm.add_group(text)
            self._refresh_lists()

    def remove_group_click(self):
        item = self.list_groups.currentItem()
        if not item: return
        name = item.data(Qt.UserRole)
        
        reply = QMessageBox.question(self, 'Remove Class', 
                                     f"Delete class '{name}' and remove all files in it?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.vm.remove_group(name)
            self._refresh_lists()
            self.list_files.clear()
            self.lbl_selected_group.setText("No class selected")
            self.btn_add_f.setEnabled(False)
            self.btn_del_f.setEnabled(False)

    def rename_group_item(self, item):
        old_name = item.text()
        # Strip stats if present? Current logic puts stats in text. 
        # Need to store raw name in UserRole or parse it.
        # Let's use UserRole for raw name.
        raw_name = item.data(Qt.UserRole)
        if not raw_name: raw_name = old_name # Fallback
        
        new_name, ok = QInputDialog.getText(self, "Rename Class", "New Name:", text=raw_name)
        if ok and new_name and new_name != raw_name:
            if new_name in self.vm.file_groups:
                QMessageBox.warning(self, "Error", "Name already exists!")
                return
            self.vm.rename_group(raw_name, new_name)
            self._refresh_lists()

    def show_group_context_menu(self, pos):
        item = self.list_groups.itemAt(pos)
        if not item: return
        
        menu = QMenu()
        act_rename = menu.addAction("Rename")
        act_color = menu.addAction("Set Color")
        act_delete = menu.addAction("Delete Class")
        
        action = menu.exec_(self.list_groups.mapToGlobal(pos))
        
        if action == act_rename:
            self.rename_group_item(item)
        elif action == act_color:
            self.set_group_color(item)
        elif action == act_delete:
            self.remove_group_click() # Will pick current item which is likely the one clicked

    def set_group_color(self, item):
        raw_name = item.data(Qt.UserRole)
        # Get current color
        curr_hex = self.vm.group_colors.get(raw_name, "#FFFFFF")
        color = QColorDialog.getColor(QColor(curr_hex), self, "Select Class Color")
        
        if color.isValid():
            self.vm.set_group_color(raw_name, color.name())
            self._refresh_lists()

    def on_group_selected(self):
        item = self.list_groups.currentItem()
        if not item:
            self.lbl_selected_group.setText("No class selected")
            self.list_files.clear()
            self.btn_add_f.setEnabled(False)
            self.btn_del_f.setEnabled(False)
            return
            
        name = item.data(Qt.UserRole) # Use raw name
        count = len(self.vm.file_groups.get(name, []))
        self.lbl_selected_group.setText(f"Class: {name} ({count} files)")
        
        # Color indicator in label
        hex_c = self.vm.group_colors.get(name, "#000000")
        self.lbl_selected_group.setStyleSheet(f"font-weight: bold; color: {hex_c};")
        
        self.btn_add_f.setEnabled(True)
        self.btn_del_f.setEnabled(True)
        
        # Refresh File List
        self.list_files.clear()
        files = self.vm.file_groups.get(name, [])
        for f in files:
            item_txt = f"{os.path.basename(f)}  [{os.path.dirname(f)}]"
            item_widget = QListWidgetItem(item_txt)
            item_widget.setData(Qt.UserRole, f)
            self.list_files.addItem(item_widget)

    # --- File Handlers ---
    def add_files_click(self):
        item = self.list_groups.currentItem()
        if not item: return
        group_name = item.data(Qt.UserRole)
        
        files, _ = QFileDialog.getOpenFileNames(self, "Select HSI Files", filter="HDR Files (*.hdr);;All Files (*)")
        if files:
            self.vm.add_files_to_group(group_name, files)

    def remove_files_click(self):
        item_g = self.list_groups.currentItem()
        if not item_g: return
        group_name = item_g.data(Qt.UserRole)
        
        items = self.list_files.selectedItems()
        if not items: return
        
        paths = [it.data(Qt.UserRole) for it in items]
        self.vm.remove_files_from_group(group_name, paths)

    # --- Reference Handlers ---
    def browse_white(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select White Ref", filter="HDR Files (*.hdr);;All Files (*)")
        if f: self.vm.set_white_ref(f)

    def browse_dark(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Dark Ref", filter="HDR Files (*.hdr);;All Files (*)")
        if f: self.vm.set_dark_ref(f)

    def _refresh_lists(self):
        # Save current selection
        curr_item = self.list_groups.currentItem()
        curr_name = curr_item.data(Qt.UserRole) if curr_item else None
        
        self.list_groups.clear()
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]
        
        for name in self.vm.file_groups.keys():
            files = self.vm.file_groups[name]
            count = len(files)
            
            # Check if ignored
            is_ignored = name.lower() in EXCLUDED_NAMES
            
            # Create Item with Icon
            display_text = f"{name} ({count})"
            if is_ignored:
                display_text = f"🚫 {display_text}"
                
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, name) # Store raw name
            
            if is_ignored:
                item.setToolTip("Excluded from Training (Ignored Class)")
                item.setForeground(QColor("gray"))
            else:
                item.setToolTip("Included in Training")
            
            # Color Icon
            hex_c = self.vm.group_colors.get(name, "#808080")
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(hex_c))
            item.setIcon(QIcon(pixmap))
            
            self.list_groups.addItem(item)
            
        # Restore selection
        if curr_name:
            # Find item with raw name
            for i in range(self.list_groups.count()):
                it = self.list_groups.item(i)
                if it.data(Qt.UserRole) == curr_name:
                    self.list_groups.setCurrentItem(it)
                    self.on_group_selected()
                    break
        else:
            self.list_files.clear()

    def _refresh_refs(self):
        self.txt_white.setText(self.vm.white_ref)
        self.txt_dark.setText(self.vm.dark_ref)

    def restore_ui(self):
        """AI가 추가함: UI 초기화 및 VM 상태 복원"""
        self._refresh_lists()
        self._refresh_refs()
        
        # Processing Mode Reset
        if hasattr(self, 'radio_ref'):
            pm = self.vm.processing_mode
            if pm == "Reflectance": self.radio_ref.setChecked(True)
            elif pm == "Absorbance": self.radio_abs.setChecked(True)
            else: self.radio_raw.setChecked(True)
