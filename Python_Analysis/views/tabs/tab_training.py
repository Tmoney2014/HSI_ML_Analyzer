from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QProgressBar, 
                             QTextEdit, QLabel, QGroupBox, QHBoxLayout, QLineEdit,
                             QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog, QSplitter,
                             QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from viewmodels.training_vm import TrainingViewModel
import os

class TabTraining(QWidget):
    def __init__(self, training_vm: TrainingViewModel):
        super().__init__()
        self.vm = training_vm
        self.init_ui()
        self.connect_signals()
        
        # AI가 수정함: VM 상태로부터 UI 초기화 (Project/Template Load 시)
        self.init_from_vm_state()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        grp_idx = QGroupBox("Training Configuration")
        # AI가 수정함: 다시 수직(VBox) 배치로 복귀하되, 간격(Spacing)을 줄임
        vbox = QVBoxLayout()
        vbox.setSpacing(2)  # 항목 간 간격을 2px로 좁힘
        vbox.setContentsMargins(5, 10, 5, 5) # 여백 조정
        
        # 1. Model Selection
        hbox_model = QHBoxLayout()
        hbox_model.addWidget(QLabel("Algorithm:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Linear SVM", "PLS-DA", "LDA"]) 
        self.combo_model.setToolTip("Select training algorithm.\nLDA/PLS-DA are recommended for spectral data.")
        hbox_model.addWidget(self.combo_model)
        vbox.addLayout(hbox_model)
        
        # 2. Validation Split Ratio
        hbox_split = QHBoxLayout()
        hbox_split.addWidget(QLabel("Validation Ratio:"))
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.05, 0.50)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(0.20)
        self.spin_ratio.setToolTip("Proportion of data used for validation (Test Set).")
        hbox_split.addWidget(self.spin_ratio)
        vbox.addLayout(hbox_split)
        

        # 3. Output Configuration
        # 3-1. Folder
        hbox_folder = QHBoxLayout()
        hbox_folder.addWidget(QLabel("Output Folder:"))
        self.txt_folder = QLineEdit("./output")
        self.txt_folder.setReadOnly(True)
        hbox_folder.addWidget(self.txt_folder)
        
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(30)
        self.btn_browse.clicked.connect(self.on_browse_click)
        hbox_folder.addWidget(self.btn_browse)
        vbox.addLayout(hbox_folder)
        
        # 3-2. Model Name
        hbox_name = QHBoxLayout()
        hbox_name.addWidget(QLabel("Model Name:"))
        self.txt_name = QLineEdit("model")
        self.txt_name.setPlaceholderText("e.g. apple_sorting_v1")
        hbox_name.addWidget(self.txt_name)
        vbox.addLayout(hbox_name)
        
        # 3-3. Description
        hbox_desc = QHBoxLayout()
        hbox_desc.addWidget(QLabel("Description:"))
        self.txt_desc = QLineEdit("")
        self.txt_desc.setPlaceholderText("Short description of the model...")
        hbox_desc.addWidget(self.txt_desc)
        vbox.addLayout(hbox_desc)
        
        # 4. Bands
        hbox_bands = QHBoxLayout()
        hbox_bands.addWidget(QLabel("Number of Bands (Features):"))
        self.spin_bands = QSpinBox()
        self.spin_bands.setRange(1, 100)
        self.spin_bands.setValue(5)
        self.spin_bands.setToolTip("Number of top spectral bands to select for the model.")
        hbox_bands.addWidget(self.spin_bands)
        vbox.addLayout(hbox_bands)
        
        grp_idx.setLayout(vbox)
        
        # AI가 수정함: 상단 설정창은 높이 고정 (늘어나지 않음)
        from PyQt5.QtWidgets import QSizePolicy
        grp_idx.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        layout.addWidget(grp_idx, 0) # Stretch = 0 (Do not expand)
        
        # Splitter (Left: Data Tree | Right: Log)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel: File Selection Tree
        left_widget = QWidget()
        vbox_left = QVBoxLayout(left_widget)
        vbox_left.setContentsMargins(0, 0, 0, 0)
        vbox_left.addWidget(QLabel("Training Data Selection:"))
        
        self.tree_files = QTreeWidget()
        self.tree_files.setHeaderLabel("Groups / Files")
        self.tree_files.itemChanged.connect(self.on_tree_item_changed) # Checkbox Handler
        vbox_left.addWidget(self.tree_files)
        
        splitter.addWidget(left_widget)
        
        # Right Panel: Log (Expanded)
        right_widget = QWidget()
        vbox_right = QVBoxLayout(right_widget)
        vbox_right.setContentsMargins(0, 0, 0, 0)
        vbox_right.addWidget(QLabel("Training Log:"))
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00FF00; font-family: Consolas; font-size: 11pt;")
        vbox_right.addWidget(self.log_text)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700]) # Ratio
        layout.addWidget(splitter, 1) # Stretch = 1 (Expand to fill bottom)
        
        # Buttons Layout (Bottom)
        hbox_btns = QHBoxLayout()
        
        self.btn_train = QPushButton("Train (Normal)")
        self.btn_train.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; font-weight: bold; height: 40px;")
        self.btn_train.clicked.connect(self.on_start_click)
        hbox_btns.addWidget(self.btn_train)
        
        self.btn_optimize = QPushButton("Find Params (No Save)") 
        self.btn_optimize.setToolTip("Run Auto-Optimization (using ONLY checked files).")
        self.btn_optimize.setStyleSheet("background-color: #9C27B0; color: white; font-size: 14px; font-weight: bold; height: 40px;")
        self.btn_optimize.clicked.connect(self.on_optimize_click)
        hbox_btns.addWidget(self.btn_optimize)
        
        layout.addLayout(hbox_btns)
        
        # AI가 추가함: UI 이벤트 연결
        self.connect_ui_events()

    def connect_signals(self):
        self.vm.log_message.connect(self.append_log)
        # self.vm.progress_update.connect(self.progress.setValue) # ProgressBar Removed
        self.vm.training_finished.connect(self.on_finished)

    def on_optimize_click(self):
        # AI가 수정함: 별도 저장 불필요 (Auto-Save 됨)
        self.log_text.clear()
        self.set_buttons_enabled(False)
        
        # 인자 없이 호출 -> VM 내부 상태 사용 (Source of Truth)
        self.vm.run_optimization()
        
    def set_buttons_enabled(self, enabled):
        self.btn_train.setEnabled(enabled)
        self.btn_optimize.setEnabled(enabled)
        self.combo_model.setEnabled(enabled)
        self.spin_ratio.setEnabled(enabled)

    def on_start_click(self):
        # AI가 수정함: 별도 저장 불필요 (Auto-Save 됨)
        self.log_text.clear()
        self.set_buttons_enabled(False)
        
        # 인자 없이 호출 -> VM 내부 상태 사용 (Source of Truth)
        self.vm.run_training()

    def on_finished(self, success):
        self.set_buttons_enabled(True)
        if success:
            # AI가 수정함: 최적화 결과의 n_features를 UI에 반영
            if hasattr(self.vm, 'best_n_features') and self.vm.best_n_features:
                self.spin_bands.setValue(self.vm.best_n_features)
                self.vm.best_n_features = None  # 1회성 사용 후 초기화
            self.append_log("Done.")
        else:
            self.append_log("Failed.")

    def init_from_vm_state(self):
        """AI가 수정함: VM 상태를 UI에 반영 (Signal 차단하여 무한루프 방지)"""
        self.blockSignals(True)
        try:
            # AI가 수정함: Folder/Name/Desc 반영
            self.txt_folder.setText(self.vm.output_folder)
            self.txt_name.setText(self.vm.model_name)
            self.txt_desc.setText(self.vm.model_desc)
            
            idx = self.combo_model.findText(self.vm.model_type)
            if idx >= 0: self.combo_model.setCurrentIndex(idx)
            
            self.spin_ratio.setValue(self.vm.val_ratio)
            self.spin_bands.setValue(self.vm.n_features)
        finally:
            self.blockSignals(False)
            
        # Refresh Tree (Data Sync)
        self.refresh_file_tree()

    def refresh_file_tree(self):
        """MainVM의 그룹/파일 정보를 트리 위젯에 표시 및 TrainingVM의 excluded 상태 반영"""
        self.tree_files.blockSignals(True)
        self.tree_files.clear()
        
        excluded = self.vm.excluded_files
        
        # 1. Groups
        for group_name, files in self.vm.main_vm.file_groups.items():
            if not files: continue
            
            grp_item = QTreeWidgetItem([group_name])
            grp_item.setFlags(grp_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            grp_item.setCheckState(0, Qt.Checked) # Default to Checked
            grp_item.setExpanded(True)
            self.tree_files.addTopLevelItem(grp_item)
            
            all_unchecked = True
            
            # 2. Files
            for f in files:
                # AI가 수정함: 상위 디렉터리 포함 (Parent/Filename)
                parent_dir = os.path.basename(os.path.dirname(f))
                fname = os.path.basename(f)
                display_name = f"{parent_dir}/{fname}"
                
                file_item = QTreeWidgetItem([display_name])
                file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                file_item.setData(0, Qt.UserRole, f) # Store full path
                
                if f in excluded:
                    file_item.setCheckState(0, Qt.Unchecked)
                else:
                    file_item.setCheckState(0, Qt.Checked)
                    all_unchecked = False
                    
                grp_item.addChild(file_item)
            
            # Update Group Check State based on children
            if all_unchecked:
                grp_item.setCheckState(0, Qt.Unchecked)
            # Tristate handles Partial automatically usually, but let's leave it to Qt
            
        self.tree_files.blockSignals(False)

    def on_tree_item_changed(self, item, column):
        """체크박스 변경 시 VM에 알림"""
        path = item.data(0, Qt.UserRole)
        if path: # It's a file
            is_checked = (item.checkState(0) == Qt.Checked)
            self.vm.set_file_excluded(path, not is_checked) # excluded = NOT checked

    def _on_ui_changed(self):
        """AI가 수정함: UI 변경 시 VM 업데이트 -> config_changed -> AutoSave"""
        self.vm.output_folder = self.txt_folder.text()
        self.vm.model_name = self.txt_name.text()
        self.vm.model_desc = self.txt_desc.text()
        self.vm.model_type = self.combo_model.currentText()
        self.vm.val_ratio = self.spin_ratio.value()
        self.vm.n_features = self.spin_bands.value()
        
        self.vm.config_changed.emit() # Notify Main Window to Save

    def connect_ui_events(self):
        """UI 변경 이벤트 연결 (init_ui 마지막에 호출)"""
        # AI가 수정함: editingFinished -> textChanged
        # txt_folder는 ReadOnly지만 Browse로 바뀜 (Browse에서 직접 emit)
        self.txt_name.textChanged.connect(self._on_ui_changed)
        self.txt_desc.textChanged.connect(self._on_ui_changed)
        self.combo_model.currentIndexChanged.connect(self._on_ui_changed)
        self.spin_ratio.valueChanged.connect(self._on_ui_changed)
        self.spin_bands.valueChanged.connect(self._on_ui_changed)

    def append_log(self, msg):
        self.log_text.append(msg)

    def append_log(self, msg):
        self.log_text.append(msg)

    def on_browse_click(self):
        """AI가 수정함: 폴더 선택 다이얼로그"""
        import os
        current_path = self.txt_folder.text()
        initial_dir = current_path if current_path else "./output"
        
        # 사용자가 "폴더"를 선택하길 원함
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            initial_dir,
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            full_path = folder_path.replace("/", "\\")
            self.txt_folder.setText(full_path)
            
            # AI가 추가함: 경로 변경 즉시 저장 (Force Sync)
            self.vm.output_folder = full_path
            self.vm.config_changed.emit()
