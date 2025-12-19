import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QListWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QListWidgetItem)
from PyQt5.QtCore import Qt
from viewmodels.main_vm import MainViewModel

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
        
        # 1. Normal Data
        grp_normal = QGroupBox("1. Normal Data (Batch)")
        vbox_n = QVBoxLayout()
        self.list_normal = QListWidget()
        self.list_normal.setSelectionMode(QListWidget.ExtendedSelection)
        vbox_n.addWidget(self.list_normal)
        
        hbox_btn_n = QHBoxLayout()
        btn_add_n = QPushButton("Add Folder/Files (+)")
        btn_add_n.clicked.connect(self.add_normal)
        btn_del_n = QPushButton("Remove Selected (-)")
        btn_del_n.clicked.connect(self.remove_normal)
        hbox_btn_n.addWidget(btn_add_n)
        hbox_btn_n.addWidget(btn_del_n)
        vbox_n.addLayout(hbox_btn_n)
        grp_normal.setLayout(vbox_n)
        layout.addWidget(grp_normal)
        
        # 2. Defect Data
        grp_defect = QGroupBox("2. Defect Data (Batch)")
        vbox_d = QVBoxLayout()
        self.list_defect = QListWidget()
        self.list_defect.setSelectionMode(QListWidget.ExtendedSelection)
        vbox_d.addWidget(self.list_defect)
        
        hbox_btn_d = QHBoxLayout()
        btn_add_d = QPushButton("Add Folder/Files (+)")
        btn_add_d.clicked.connect(self.add_defect)
        btn_del_d = QPushButton("Remove Selected (-)")
        btn_del_d.clicked.connect(self.remove_defect)
        hbox_btn_d.addWidget(btn_add_d)
        hbox_btn_d.addWidget(btn_del_d)
        vbox_d.addLayout(hbox_btn_d)
        grp_defect.setLayout(vbox_d)
        layout.addWidget(grp_defect)
        
        # 3. References
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
        self.radio_raw = QRadioButton("Raw Data Mode (DN)")
        self.radio_ref = QRadioButton("Reflectance Mode")
        self.radio_raw.setChecked(True)
        
        # Group for exclusive checking
        self.bg = QButtonGroup(self)
        self.bg.addButton(self.radio_raw)
        self.bg.addButton(self.radio_ref)
        
        self.radio_ref.toggled.connect(self.on_mode_change)
        
        vbox_m.addWidget(self.radio_raw)
        vbox_m.addWidget(self.radio_ref)
        grp_mode.setLayout(vbox_m)
        vbox_r.addWidget(grp_mode)
        
        vbox_r.addStretch()
        grp_ref.setLayout(vbox_r)
        layout.addWidget(grp_ref)
        
    def on_mode_change(self, checked):
        # Only handle if ref is checked (True)
        if self.radio_ref.isChecked():
            # Validate White Reference
            if not self.vm.white_ref or not os.path.exists(self.vm.white_ref):
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", "Cannot switch to Reflectance Mode!\nPlease load a White Reference file first.")
                # Revert UI
                self.radio_raw.setChecked(True)
                return
            
            self.vm.set_use_ref(True)
        else:
            self.vm.set_use_ref(False)

    def add_normal(self):
        files = self._browse_files()
        if files:
            self.vm.add_normal_files(files)

    def remove_normal(self):
        self._remove_from_list(self.list_normal, is_normal=True)

    def add_defect(self):
        files = self._browse_files()
        if files:
            self.vm.add_defect_files(files)

    def remove_defect(self):
        self._remove_from_list(self.list_defect, is_normal=False)

    def browse_white(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select White Ref", filter="HDR Files (*.hdr);;All Files (*)")
        if f:
            self.vm.set_white_ref(f)

    def browse_dark(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Dark Ref", filter="HDR Files (*.hdr);;All Files (*)")
        if f:
            self.vm.set_dark_ref(f)

    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select HSI Files", filter="HDR Files (*.hdr);;All Files (*)")
        return files

    def _remove_from_list(self, list_widget, is_normal):
        items = list_widget.selectedItems()
        if not items: return
        
        # We need to remove from VM list. 
        # Map selected UI items to indices or paths.
        # Since logic might be complex with indices if filtering (not using filtering here),
        # simpler to reconstruct the list.
        
        paths_to_remove = set()
        for item in items:
            # Assuming format "basename [full_path]" -> extract full path?
            # Actually, UserRole is best practice but let's see how I populate it.
            # If I didn't set UserRole, I rely on string parsing or sync.
            # In _refresh_lists, I'll set UserRole.
            path = item.data(Qt.UserRole)
            if path: paths_to_remove.add(path)
            
        current_list = self.vm.normal_files if is_normal else self.vm.defect_files
        new_list = [f for f in current_list if f not in paths_to_remove]
        
        if is_normal:
            self.vm.normal_files = new_list # Direct assign? Need a setter/method in VM to emit signal?
            # MainVM likely doesn't have a specific 'replace list' method but modifying attribute directly is bad practice if validation needed.
            # But currently python properties are not used for normal_files list.
            # Best to update the list and emit signal manually or add a method.
            # Let's verify MainVM has files_changed emission on modification.
            # It only has `add_normal_files` and `clear`.
            # I will modify list and emit signal manually for now since I can access signal.
            self.vm.normal_files = new_list
            self.vm.files_changed.emit()
        else:
            self.vm.defect_files = new_list
            self.vm.files_changed.emit()

    def _refresh_lists(self):
        from PyQt5.QtCore import Qt # Import needed here or top level
        
        self.list_normal.clear()
        for f in self.vm.normal_files:
            item_txt = f"{os.path.basename(f)}  [{os.path.dirname(f)}]"
            item = os.path.basename(f) if len(f) > 50 else item_txt # Adaptive logic or just full?
            from PyQt5.QtWidgets import QListWidgetItem
            item_widget = QListWidgetItem(item_txt)
            item_widget.setData(Qt.UserRole, f)
            self.list_normal.addItem(item_widget)
            
        self.list_defect.clear()
        for f in self.vm.defect_files:
            item_txt = f"{os.path.basename(f)}  [{os.path.dirname(f)}]"
            item_widget = QListWidgetItem(item_txt)
            item_widget.setData(Qt.UserRole, f)
            self.list_defect.addItem(item_widget)

    def _refresh_refs(self):
        self.txt_white.setText(self.vm.white_ref)
        self.txt_dark.setText(self.vm.dark_ref)
