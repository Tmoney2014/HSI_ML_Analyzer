from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QProgressBar, 
                             QTextEdit, QLabel, QGroupBox, QHBoxLayout, QLineEdit,
                             QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog, QSplitter,
                             QTreeWidget, QTreeWidgetItem, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from viewmodels.training_vm import TrainingViewModel
import os

class TabTraining(QWidget):
    _EXPERIMENT_BAND_METHOD_OPTIONS = [  # AI가 수정함: Export Matrix용 밴드 선택 옵션
        ("SPA", "spa"),
        ("ANOVA-F", "anova_f"),
        ("SPA-LDA Fast", "spa_lda_fast"),
        ("SPA-LDA Greedy", "spa_lda_greedy"),
        ("LDA-coef", "lda_coef"),
        ("Full Band", "full"),
    ]
    _EXPERIMENT_MODEL_OPTIONS = [  # AI가 수정함: Export Matrix용 모델 옵션
        "LDA",
        "Linear SVM",
        "PLS-DA",
        "Ridge Classifier",
        "Logistic Regression",
    ]

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
        self.combo_model.addItems(["Linear SVM", "PLS-DA", "LDA", "Ridge Classifier", "Logistic Regression"])  # AI가 수정함: Ridge Classifier, Logistic Regression 추가
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
        
        # AI가 수정함: Band Selection Method 선택 UI
        hbox_band_method = QHBoxLayout()
        hbox_band_method.addWidget(QLabel("Band Selection:"))
        self.combo_band_method = QComboBox()
        self.combo_band_method.addItems(["SPA", "Full Band", "ANOVA-F", "SPA-LDA Fast", "SPA-LDA Greedy", "LDA-coef"])  # AI가 수정함: supervised 방법 4개 추가
        self.combo_band_method.setToolTip("SPA: Successive Projections (orthogonal), Full Band: Use all valid bands (no selection)")  # AI가 수정함: 툴팁 업데이트
        hbox_band_method.addWidget(self.combo_band_method)
        vbox.addLayout(hbox_band_method)

        grp_experiment = QGroupBox("Experiment Matrix Selection")  # AI가 수정함: Export Matrix 전용 다중 선택 UI
        exp_vbox = QVBoxLayout()  # AI가 수정함: Experiment Matrix 그룹 레이아웃
        exp_vbox.setSpacing(4)  # AI가 수정함: 그룹 내부 간격 축소
        exp_vbox.addWidget(QLabel("Choose multiple band methods and models for paper-style matrix export."))  # AI가 수정함: 설명 라벨

        exp_hbox = QHBoxLayout()  # AI가 수정함: 두 개의 리스트를 나란히 배치

        exp_band_vbox = QVBoxLayout()  # AI가 수정함: 밴드 방법 리스트 컬럼
        exp_band_vbox.addWidget(QLabel("Band Methods:"))  # AI가 수정함:
        self.list_experiment_band_methods = QListWidget()  # AI가 수정함: Export Matrix 밴드 방법 체크 리스트
        self.list_experiment_band_methods.setToolTip("Checked methods will be combined with checked models during Export Matrix.")  # AI가 수정함:
        for display_name, method_key in self._EXPERIMENT_BAND_METHOD_OPTIONS:  # AI가 수정함: 체크 가능한 항목 생성
            item = QListWidgetItem(display_name)  # AI가 수정함: 표시 이름
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)  # AI가 수정함: 체크 가능 항목
            item.setData(Qt.UserRole, method_key)  # AI가 수정함: canonical method key 저장
            item.setCheckState(Qt.Unchecked)  # AI가 수정함: init_from_vm_state에서 선택 복원
            self.list_experiment_band_methods.addItem(item)  # AI가 수정함: 리스트에 추가
        exp_band_vbox.addWidget(self.list_experiment_band_methods)  # AI가 수정함:
        exp_hbox.addLayout(exp_band_vbox)  # AI가 수정함:

        exp_model_vbox = QVBoxLayout()  # AI가 수정함: 모델 리스트 컬럼
        exp_model_vbox.addWidget(QLabel("Models:"))  # AI가 수정함:
        self.list_experiment_models = QListWidget()  # AI가 수정함: Export Matrix 모델 체크 리스트
        self.list_experiment_models.setToolTip("Checked models will be paired with each checked band method during Export Matrix.")  # AI가 수정함:
        for model_name in self._EXPERIMENT_MODEL_OPTIONS:  # AI가 수정함: 체크 가능한 모델 항목 생성
            item = QListWidgetItem(model_name)  # AI가 수정함: 표시 이름 == canonical model name
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)  # AI가 수정함: 체크 가능 항목
            item.setData(Qt.UserRole, model_name)  # AI가 수정함: canonical model name 저장
            item.setCheckState(Qt.Unchecked)  # AI가 수정함: init_from_vm_state에서 선택 복원
            self.list_experiment_models.addItem(item)  # AI가 수정함: 리스트에 추가
        exp_model_vbox.addWidget(self.list_experiment_models)  # AI가 수정함:
        exp_hbox.addLayout(exp_model_vbox)  # AI가 수정함:

        exp_vbox.addLayout(exp_hbox)  # AI가 수정함: 좌우 리스트 레이아웃 추가
        grp_experiment.setLayout(exp_vbox)  # AI가 수정함: 그룹 박스 레이아웃 설정
        vbox.addWidget(grp_experiment)  # AI가 수정함: Training Configuration 그룹에 포함
        
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
        
        self.progress_bar = QProgressBar()  # AI가 수정함: Progress Bar 복원
        self.progress_bar.setRange(0, 100)   # AI가 수정함:
        self.progress_bar.setValue(0)         # AI가 수정함:
        self.progress_bar.setTextVisible(True)  # AI가 수정함:
        layout.addWidget(self.progress_bar)  # AI가 수정함: splitter 아래, 버튼 위
        
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

        hbox_btns2 = QHBoxLayout()  # AI가 수정함: 두 번째 버튼 행
        self.btn_export = QPushButton("Export Matrix")  # AI가 수정함: Export 버튼
        self.btn_export.setToolTip("Run experiment grid (band_methods × models) and export CSV + confusion matrix")
        self.btn_export.setStyleSheet("background-color: #FF9800; color: white; font-size: 13px; font-weight: bold; height: 36px;")
        self.btn_export.clicked.connect(self.on_export_click)
        hbox_btns2.addWidget(self.btn_export)

        self.btn_open_folder = QPushButton("Open Experiments Folder")  # AI가 수정함: 폴더 열기 버튼
        self.btn_open_folder.setStyleSheet("background-color: #607D8B; color: white; font-size: 13px; height: 36px;")
        self.btn_open_folder.clicked.connect(self.on_open_folder_click)
        hbox_btns2.addWidget(self.btn_open_folder)
        layout.addLayout(hbox_btns2)
        
        # AI가 추가함: UI 이벤트 연결
        self.connect_ui_events()

    def connect_signals(self):
        self.vm.log_message.connect(self.append_log)
        self.vm.progress_update.connect(self.progress_bar.setValue)  # AI가 수정함: Progress Bar 연결
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
        self.btn_export.setEnabled(enabled)      # AI가 수정함: Export 버튼
        self.btn_open_folder.setEnabled(enabled) # AI가 수정함: Open Folder 버튼
        self.combo_model.setEnabled(enabled)
        self.spin_ratio.setEnabled(enabled)
        self.combo_band_method.setEnabled(enabled)  # AI가 수정함: 훈련 중 비활성화
        self.list_experiment_band_methods.setEnabled(enabled)  # AI가 수정함: Export Matrix 밴드 리스트 비활성화 포함
        self.list_experiment_models.setEnabled(enabled)  # AI가 수정함: Export Matrix 모델 리스트 비활성화 포함
        # AI가 수정함: Full Band 모드에서는 훈련 완료 후에도 spin_bands 비활성화 유지
        if enabled:  # AI가 수정함:
            self.spin_bands.setEnabled(self.vm.band_selection_method != "full")
        else:  # AI가 수정함:
            self.spin_bands.setEnabled(False)  # AI가 수정함:

    def on_start_click(self):
        # AI가 수정함: 별도 저장 불필요 (Auto-Save 됨)
        self.log_text.clear()
        self.set_buttons_enabled(False)
        
        # 인자 없이 호출 -> VM 내부 상태 사용 (Source of Truth)
        self.vm.run_training()

    def on_finished(self, success):
        self.progress_bar.setValue(0)  # AI가 수정함: 완료 후 progress bar 초기화
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
            # AI가 수정함: Band Selection Method 복원
            _method_display = {"spa": "SPA", "full": "Full Band", "anova_f": "ANOVA-F", "spa_lda_fast": "SPA-LDA Fast", "spa_lda_greedy": "SPA-LDA Greedy", "lda_coef": "LDA-coef"}  # AI가 수정함: supervised 방법 4개 추가
            idx = self.combo_band_method.findText(_method_display.get(self.vm.band_selection_method, "SPA"))
            if idx >= 0: self.combo_band_method.setCurrentIndex(idx)
            self._restore_checked_items(self.list_experiment_band_methods, self.vm.experiment_band_methods)  # AI가 수정함: Export Matrix 밴드 선택 복원
            self._restore_checked_items(self.list_experiment_models, self.vm.experiment_model_types)  # AI가 수정함: Export Matrix 모델 선택 복원
        finally:
            self.blockSignals(False)
        
        # AI가 수정함: Full Band 모드에서 Number of Bands 스핀박스 비활성화
        self.spin_bands.setEnabled(self.vm.band_selection_method != "full")

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
        # AI가 수정함: Band Selection Method VM 동기화
        _method_map = {"SPA": "spa", "Full Band": "full", "ANOVA-F": "anova_f", "SPA-LDA Fast": "spa_lda_fast", "SPA-LDA Greedy": "spa_lda_greedy", "LDA-coef": "lda_coef"}  # AI가 수정함: supervised 방법 4개 추가
        self.vm.band_selection_method = _method_map.get(self.combo_band_method.currentText(), "spa")
        # AI가 수정함: Full Band 모드에서 Number of Bands 스핀박스 비활성화
        self.spin_bands.setEnabled(self.vm.band_selection_method != "full")
        
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
        self.combo_band_method.currentIndexChanged.connect(self._on_ui_changed)  # AI가 수정함
        self.list_experiment_band_methods.itemChanged.connect(self._on_experiment_selection_changed)  # AI가 수정함: Export Matrix 밴드 체크 변경
        self.list_experiment_models.itemChanged.connect(self._on_experiment_selection_changed)  # AI가 수정함: Export Matrix 모델 체크 변경

    def _restore_checked_items(self, list_widget, selected_values):  # AI가 수정함: 체크 리스트 선택 복원 헬퍼
        selected_set = set(selected_values or [])  # AI가 수정함: membership lookup 최적화
        list_widget.blockSignals(True)  # AI가 수정함: 복원 중 itemChanged 루프 방지
        try:
            for index in range(list_widget.count()):  # AI가 수정함: 전체 항목 순회
                item = list_widget.item(index)  # AI가 수정함:
                item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in selected_set else Qt.Unchecked)  # AI가 수정함: 저장된 선택값 반영
        finally:
            list_widget.blockSignals(False)  # AI가 수정함:

    def _get_checked_values(self, list_widget):  # AI가 수정함: 체크된 canonical 값 목록 수집
        values = []  # AI가 수정함: 반환 리스트 초기화
        for index in range(list_widget.count()):  # AI가 수정함: 전체 항목 순회
            item = list_widget.item(index)  # AI가 수정함:
            if item.checkState() == Qt.Checked:  # AI가 수정함: 체크된 항목만 수집
                values.append(item.data(Qt.UserRole))  # AI가 수정함: canonical 값 사용
        return values  # AI가 수정함: 선택값 반환

    def _on_experiment_selection_changed(self, _item):  # AI가 수정함: Export Matrix 체크 상태를 VM에 저장
        self.vm.experiment_band_methods = self._get_checked_values(self.list_experiment_band_methods)  # AI가 수정함: 밴드 선택 동기화
        self.vm.experiment_model_types = self._get_checked_values(self.list_experiment_models)  # AI가 수정함: 모델 선택 동기화
        self.vm.config_changed.emit()  # AI가 수정함: 프로젝트 저장 반영

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

    def on_export_click(self):  # AI가 수정함: Export Matrix 버튼 핸들러
        """현재 UI 체크박스 기준으로 모든 활성 band_method × model 조합 실험 그리드 실행"""
        band_methods = self._get_checked_values(self.list_experiment_band_methods)  # AI가 수정함: 체크된 밴드 방법 전체 수집
        model_types = self._get_checked_values(self.list_experiment_models)  # AI가 수정함: 체크된 모델 전체 수집
        if not band_methods:  # AI가 수정함: 최소 1개 밴드 방법 필수
            self.append_log("⚠️ Select at least one band method for Export Matrix.")
            return
        if not model_types:  # AI가 수정함: 최소 1개 모델 필수
            self.append_log("⚠️ Select at least one model for Export Matrix.")
            return
        self.log_text.clear()
        self.set_buttons_enabled(False)
        self.vm.run_experiment_grid(band_methods, model_types)

    def on_open_folder_click(self):  # AI가 수정함: 실험 폴더 열기
        import os, subprocess
        folder = os.path.join(self.vm.output_folder, "experiments")  # AI가 수정함: 실제 실험 결과 서브폴더 열기
        if os.path.exists(folder):
            subprocess.Popen(f'explorer "{folder}"')  # AI가 수정함: Windows 탐색기 열기
        else:
            self.append_log(f"⚠️ Folder not found: {folder}")

