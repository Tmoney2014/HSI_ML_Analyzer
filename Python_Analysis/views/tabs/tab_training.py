from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QProgressBar, QTextEdit, QLabel,
    QGroupBox, QHBoxLayout, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QSplitter, QTreeWidget, QTreeWidgetItem, QListWidget,
    QListWidgetItem, QFormLayout, QToolButton
)
from PyQt5.QtCore import Qt
from viewmodels.training_vm import TrainingViewModel
import os


class TabTraining(QWidget):
    _EXPERIMENT_BAND_METHOD_OPTIONS = [
        ("SPA", "spa"),
        ("ANOVA-F", "anova_f"),
        ("SPA-LDA Fast", "spa_lda_fast"),
        ("SPA-LDA Greedy", "spa_lda_greedy"),
        ("LDA-coef", "lda_coef"),
        ("Full Band", "full"),
    ]
    _EXPERIMENT_MODEL_OPTIONS = [
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
        self.init_from_vm_state()

    def init_ui(self):
        layout = QVBoxLayout(self)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        # Left panel: configuration + actions
        left_widget = QWidget()
        left_widget.setMinimumWidth(360)
        left_widget.setMaximumWidth(460)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        grp_quick = QGroupBox("1. Quick Run")
        quick_layout = QVBoxLayout()
        quick_layout.setContentsMargins(8, 10, 8, 8)
        quick_layout.setSpacing(6)
        lbl_quick_help = QLabel("Choose the main recipe used by Train and Find Best Params.")
        lbl_quick_help.setStyleSheet("color: #666; font-size: 11px;")
        lbl_quick_help.setWordWrap(True)
        quick_layout.addWidget(lbl_quick_help)

        form_run = QFormLayout()
        form_run.setContentsMargins(0, 0, 0, 0)
        form_run.setSpacing(6)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["Linear SVM", "PLS-DA", "LDA", "Ridge Classifier", "Logistic Regression"])
        self.combo_model.setToolTip("Select training algorithm. LDA/PLS-DA are recommended for spectral data.")
        form_run.addRow("Algorithm", self.combo_model)

        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.05, 0.50)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(0.20)
        self.spin_ratio.setToolTip("Proportion of data used for validation (test set).")
        form_run.addRow("Validation Ratio", self.spin_ratio)

        self.spin_bands = QSpinBox()
        self.spin_bands.setRange(1, 100)
        self.spin_bands.setValue(5)
        self.spin_bands.setToolTip("Number of top spectral bands to select for the model.")
        form_run.addRow("Number of Bands", self.spin_bands)

        self.combo_band_method = QComboBox()
        self.combo_band_method.addItems(["SPA", "Full Band", "ANOVA-F", "SPA-LDA Fast", "SPA-LDA Greedy", "LDA-coef"])
        self.combo_band_method.setToolTip("Band selection used by Train / Find Best Params. Export Matrix uses its own multi-select list below.")
        form_run.addRow("Band Selection", self.combo_band_method)

        quick_layout.addLayout(form_run)
        grp_quick.setLayout(quick_layout)
        left_layout.addWidget(grp_quick)

        grp_output = QGroupBox("2. Output & Metadata")
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(8, 10, 8, 8)
        output_layout.setSpacing(6)
        lbl_output_help = QLabel("Choose where outputs are saved and add optional export metadata.")
        lbl_output_help.setStyleSheet("color: #666; font-size: 11px;")
        lbl_output_help.setWordWrap(True)
        output_layout.addWidget(lbl_output_help)

        form_output = QFormLayout()
        form_output.setContentsMargins(0, 0, 0, 0)
        form_output.setSpacing(6)

        self.txt_folder = QLineEdit("./output")
        self.txt_folder.setReadOnly(True)
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(30)
        self.btn_browse.clicked.connect(self.on_browse_click)

        folder_row = QWidget()
        folder_row_layout = QHBoxLayout(folder_row)
        folder_row_layout.setContentsMargins(0, 0, 0, 0)
        folder_row_layout.setSpacing(6)
        folder_row_layout.addWidget(self.txt_folder)
        folder_row_layout.addWidget(self.btn_browse)
        form_output.addRow("Output Folder", folder_row)

        self.txt_name = QLineEdit("model")
        self.txt_name.setPlaceholderText("e.g. apple_sorting_v1")
        form_output.addRow("Model Name", self.txt_name)

        self.txt_desc = QLineEdit("")
        self.txt_desc.setPlaceholderText("Short description of the model...")
        form_output.addRow("Description", self.txt_desc)

        output_layout.addLayout(form_output)
        grp_output.setLayout(output_layout)
        left_layout.addWidget(grp_output)

        grp_actions = QGroupBox("3. Run")
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(8, 10, 8, 8)
        actions_layout.setSpacing(8)
        lbl_actions_help = QLabel("Train the current recipe, or search for a better recipe before saving.")
        lbl_actions_help.setStyleSheet("color: #666; font-size: 11px;")
        lbl_actions_help.setWordWrap(True)
        actions_layout.addWidget(lbl_actions_help)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        actions_layout.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; font-weight: bold; height: 38px;")
        self.btn_train.clicked.connect(self.on_start_click)
        action_row.addWidget(self.btn_train)

        self.btn_optimize = QPushButton("Find Best Params")
        self.btn_optimize.setToolTip("Run Auto-Optimization using only checked files.")
        self.btn_optimize.setStyleSheet("background-color: #9C27B0; color: white; font-size: 14px; font-weight: bold; height: 38px;")
        self.btn_optimize.clicked.connect(self.on_optimize_click)
        action_row.addWidget(self.btn_optimize)
        actions_layout.addLayout(action_row)

        self.lbl_spa_greedy_warning = QLabel("⚠️ SPA-LDA Greedy is very slow — optimization may take a long time.")
        self.lbl_spa_greedy_warning.setStyleSheet("color: #FF6F00; font-size: 11px;")
        self.lbl_spa_greedy_warning.setWordWrap(True)
        self.lbl_spa_greedy_warning.setVisible(False)  # AI가 수정함: 기본 숨김
        actions_layout.addWidget(self.lbl_spa_greedy_warning)

        self.lbl_action_summary = QLabel("Tip: Train runs one recipe. Find Best Params updates the suggested band count.")
        self.lbl_action_summary.setStyleSheet("color: #666; font-size: 11px;")
        self.lbl_action_summary.setWordWrap(True)
        actions_layout.addWidget(self.lbl_action_summary)
        grp_actions.setLayout(actions_layout)
        left_layout.addWidget(grp_actions)

        grp_experiment = QGroupBox("4. Experiment Matrix (Advanced)")
        exp_outer = QVBoxLayout()
        exp_outer.setContentsMargins(8, 10, 8, 8)
        exp_outer.setSpacing(6)

        self.lbl_experiment_summary = QLabel("")
        self.lbl_experiment_summary.setStyleSheet("color: #666; font-size: 11px;")
        self.lbl_experiment_summary.setWordWrap(True)
        exp_outer.addWidget(self.lbl_experiment_summary)

        self.btn_toggle_experiment = QToolButton()
        self.btn_toggle_experiment.setCheckable(True)
        self.btn_toggle_experiment.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_toggle_experiment.clicked.connect(self._toggle_experiment_section)
        exp_outer.addWidget(self.btn_toggle_experiment, 0, Qt.AlignLeft)

        self.experiment_content = QWidget()
        exp_layout = QVBoxLayout(self.experiment_content)
        exp_layout.setContentsMargins(0, 0, 0, 0)
        exp_layout.setSpacing(6)
        lbl_exp_help = QLabel("Choose multiple band methods and models. Export will create one aggregate CSV, paper-ready summary tables, heatmaps, and confusion matrix images.")
        lbl_exp_help.setStyleSheet("color: #666; font-size: 11px;")
        lbl_exp_help.setWordWrap(True)
        exp_layout.addWidget(lbl_exp_help)

        exp_lists = QHBoxLayout()

        exp_band_vbox = QVBoxLayout()
        exp_band_vbox.addWidget(QLabel("Band Methods"))
        self.list_experiment_band_methods = QListWidget()
        self.list_experiment_band_methods.setToolTip("Checked methods will be combined with checked models during Export Matrix.")
        for display_name, method_key in self._EXPERIMENT_BAND_METHOD_OPTIONS:
            item = QListWidgetItem(display_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, method_key)
            item.setCheckState(Qt.Unchecked)
            self.list_experiment_band_methods.addItem(item)
        exp_band_vbox.addWidget(self.list_experiment_band_methods)
        exp_lists.addLayout(exp_band_vbox)

        exp_model_vbox = QVBoxLayout()
        exp_model_vbox.addWidget(QLabel("Models"))
        self.list_experiment_models = QListWidget()
        self.list_experiment_models.setToolTip("Checked models will be paired with each checked band method during Export Matrix.")
        for model_name in self._EXPERIMENT_MODEL_OPTIONS:
            item = QListWidgetItem(model_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, model_name)
            item.setCheckState(Qt.Unchecked)
            self.list_experiment_models.addItem(item)
        exp_model_vbox.addWidget(self.list_experiment_models)
        exp_nbands_vbox = QVBoxLayout()
        exp_nbands_vbox.addWidget(QLabel("Bands (n)"))
        self.list_experiment_n_bands = QListWidget()
        self.list_experiment_n_bands.setToolTip("Check the band counts to include in the experiment grid.")
        self.list_experiment_n_bands.setMaximumHeight(130)
        for n in [5, 10, 15, 20, 25, 30, 35, 40]:
            item = QListWidgetItem(str(n))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, n)
            item.setCheckState(Qt.Unchecked)
            self.list_experiment_n_bands.addItem(item)
        exp_nbands_vbox.addWidget(self.list_experiment_n_bands)
        exp_lists.addLayout(exp_nbands_vbox)

        exp_layout.addLayout(exp_lists)

        # AI가 수정함: Gap 범위 설정 위젯
        exp_gap_row = QHBoxLayout()
        exp_gap_row.addWidget(QLabel("Gap range:"))
        self.spin_experiment_gap_min = QSpinBox()
        self.spin_experiment_gap_min.setRange(1, 40)
        self.spin_experiment_gap_min.setValue(1)
        self.spin_experiment_gap_min.setToolTip("Minimum SimpleDeriv gap value for grid search.")
        exp_gap_row.addWidget(self.spin_experiment_gap_min)
        exp_gap_row.addWidget(QLabel("–"))
        self.spin_experiment_gap_max = QSpinBox()
        self.spin_experiment_gap_max.setRange(1, 40)
        self.spin_experiment_gap_max.setValue(20)
        self.spin_experiment_gap_max.setToolTip("Maximum SimpleDeriv gap value for grid search.")
        exp_gap_row.addWidget(self.spin_experiment_gap_max)
        exp_gap_row.addStretch(1)
        exp_layout.addLayout(exp_gap_row)

        self.lbl_experiment_spa_greedy_warning = QLabel("⚠️ SPA-LDA Greedy selected — experiment may be very slow.")
        self.lbl_experiment_spa_greedy_warning.setStyleSheet("color: #FF6F00; font-size: 11px;")
        self.lbl_experiment_spa_greedy_warning.setWordWrap(True)
        self.lbl_experiment_spa_greedy_warning.setVisible(False)  # AI가 수정함: 기본 숨김
        exp_layout.addWidget(self.lbl_experiment_spa_greedy_warning)

        self.lbl_experiment_gap_disabled = QLabel("ℹ️ Gap search disabled — SimpleDeriv not in current prep chain.")
        self.lbl_experiment_gap_disabled.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_experiment_gap_disabled.setWordWrap(True)
        self.lbl_experiment_gap_disabled.setVisible(False)
        exp_layout.addWidget(self.lbl_experiment_gap_disabled)

        exp_actions = QHBoxLayout()
        self.btn_export = QPushButton("Export Paper Summary")
        self.btn_export.setToolTip("Run experiment grid and export one aggregate CSV, paper summary tables, heatmaps, and confusion matrices")
        self.btn_export.setStyleSheet("background-color: #FF9800; color: white; font-size: 13px; font-weight: bold; height: 34px;")
        self.btn_export.clicked.connect(self.on_export_click)
        exp_actions.addWidget(self.btn_export)

        self.btn_open_folder = QPushButton("Open Results Folder")
        self.btn_open_folder.setStyleSheet("background-color: #607D8B; color: white; font-size: 13px; height: 34px;")
        self.btn_open_folder.clicked.connect(self.on_open_folder_click)
        exp_actions.addWidget(self.btn_open_folder)
        exp_layout.addLayout(exp_actions)

        exp_outer.addWidget(self.experiment_content)
        grp_experiment.setLayout(exp_outer)
        left_layout.addWidget(grp_experiment)
        left_layout.addStretch(1)

        main_splitter.addWidget(left_widget)

        # Right panel: data tree + log
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)

        data_group = QGroupBox("Training Data Selection")
        data_layout = QVBoxLayout()
        data_layout.setContentsMargins(8, 10, 8, 8)
        data_layout.setSpacing(6)
        lbl_data_help = QLabel("Uncheck files you want to exclude from training and optimization.")
        lbl_data_help.setStyleSheet("color: #666; font-size: 11px;")
        lbl_data_help.setWordWrap(True)
        data_layout.addWidget(lbl_data_help)

        self.tree_files = QTreeWidget()
        self.tree_files.setHeaderLabel("Groups / Files")
        self.tree_files.itemChanged.connect(self.on_tree_item_changed)
        data_layout.addWidget(self.tree_files)
        data_group.setLayout(data_layout)
        right_splitter.addWidget(data_group)

        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(8, 10, 8, 8)
        log_layout.setSpacing(6)
        lbl_log_help = QLabel("Training, optimization, and experiment messages appear here.")
        lbl_log_help.setStyleSheet("color: #666; font-size: 11px;")
        lbl_log_help.setWordWrap(True)
        log_layout.addWidget(lbl_log_help)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00FF00; font-family: Consolas; font-size: 11pt;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_splitter.addWidget(log_group)
        right_splitter.setSizes([360, 320])

        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([390, 860])
        layout.addWidget(main_splitter, 1)

        self.experiment_content.setVisible(False)
        self._update_experiment_summary()
        self._toggle_experiment_section(False)
        self._update_gap_widgets_enabled()

        self.connect_ui_events()

    def connect_signals(self):
        self.vm.log_message.connect(self.append_log)
        self.vm.progress_update.connect(self.progress_bar.setValue)
        self.vm.training_finished.connect(self.on_finished)

    def on_optimize_click(self):
        self.log_text.clear()
        self.set_buttons_enabled(False)
        self.vm.run_optimization()

    def set_buttons_enabled(self, enabled):
        self.btn_train.setEnabled(enabled)
        self.btn_optimize.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.btn_open_folder.setEnabled(enabled)
        self.btn_toggle_experiment.setEnabled(enabled)
        self.combo_model.setEnabled(enabled)
        self.spin_ratio.setEnabled(enabled)
        self.combo_band_method.setEnabled(enabled)
        self.txt_name.setEnabled(enabled)
        self.txt_desc.setEnabled(enabled)
        self.btn_browse.setEnabled(enabled)
        self.list_experiment_band_methods.setEnabled(enabled)
        self.list_experiment_models.setEnabled(enabled)
        self.list_experiment_n_bands.setEnabled(enabled)
        if enabled:
            self.spin_bands.setEnabled(self.vm.band_selection_method != "full")
            self._update_gap_widgets_enabled()
        else:
            self.spin_bands.setEnabled(False)
            self.spin_experiment_gap_min.setEnabled(False)
            self.spin_experiment_gap_max.setEnabled(False)

    def on_start_click(self):
        self.log_text.clear()
        self.set_buttons_enabled(False)
        self.vm.run_training()

    def on_finished(self, success):
        self.progress_bar.setValue(0)
        self.set_buttons_enabled(True)
        if success:
            if hasattr(self.vm, 'best_n_features') and self.vm.best_n_features and getattr(self.vm, 'band_selection_method', None) != 'full':  # AI가 수정함: full 모드에서는 spin_bands 갱신 건너뜀
                self.spin_bands.setValue(self.vm.best_n_features)
                self.vm.best_n_features = None
            if hasattr(self.vm, 'best_band_method') and self.vm.best_band_method:  # AI가 추가함: 최적 band method UI 반영
                method_display = {
                    "spa": "SPA", "full": "Full Band", "anova_f": "ANOVA-F",
                    "spa_lda_fast": "SPA-LDA Fast", "spa_lda_greedy": "SPA-LDA Greedy", "lda_coef": "LDA-coef",
                }
                display_text = method_display.get(self.vm.best_band_method, "SPA")
                idx = self.combo_band_method.findText(display_text)
                if idx >= 0:
                    self.combo_band_method.setCurrentIndex(idx)  # AI가 수정함: combo 업데이트 (signal 발동 → _on_ui_changed 연쇄)
                self.vm.best_band_method = None  # AI가 수정함: transient 값 소비 (일회성)
            self.append_log("Done.")
        else:
            self.append_log("Failed.")

    def init_from_vm_state(self):
        self._set_form_signal_blocking(True)
        try:
            self.txt_folder.setText(self.vm.output_folder)
            self.txt_name.setText(self.vm.model_name)
            self.txt_desc.setText(self.vm.model_desc)

            idx = self.combo_model.findText(self.vm.model_type)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)

            self.spin_ratio.setValue(self.vm.val_ratio)
            self.spin_bands.setValue(self.vm.n_features)
            method_display = {
                "spa": "SPA",
                "full": "Full Band",
                "anova_f": "ANOVA-F",
                "spa_lda_fast": "SPA-LDA Fast",
                "spa_lda_greedy": "SPA-LDA Greedy",
                "lda_coef": "LDA-coef",
            }
            idx = self.combo_band_method.findText(method_display.get(self.vm.band_selection_method, "SPA"))
            if idx >= 0:
                self.combo_band_method.setCurrentIndex(idx)
            self._restore_checked_items(self.list_experiment_band_methods, self.vm.experiment_band_methods)
            self._restore_checked_items(self.list_experiment_models, self.vm.experiment_model_types)
            self._restore_checked_n_bands(getattr(self.vm, 'experiment_n_bands_list', []))
            _spin_blocked = [self.spin_experiment_gap_min, self.spin_experiment_gap_max]
            for w in _spin_blocked: w.blockSignals(True)
            self.spin_experiment_gap_min.setValue(getattr(self.vm, 'experiment_gap_min', 1))
            self.spin_experiment_gap_max.setValue(getattr(self.vm, 'experiment_gap_max', 20))
            for w in _spin_blocked: w.blockSignals(False)
        finally:
            self._set_form_signal_blocking(False)

        self.spin_bands.setEnabled(self.vm.band_selection_method != "full")
        self._update_experiment_summary()
        self.refresh_file_tree()

    def refresh_file_tree(self):
        self.tree_files.blockSignals(True)
        self.tree_files.clear()

        excluded = self.vm.excluded_files
        for group_name, files in self.vm.main_vm.file_groups.items():
            if not files:
                continue

            grp_item = QTreeWidgetItem([group_name])
            grp_item.setFlags(grp_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            grp_item.setCheckState(0, Qt.Checked)
            grp_item.setExpanded(True)
            self.tree_files.addTopLevelItem(grp_item)

            all_unchecked = True
            for f in files:
                parent_dir = os.path.basename(os.path.dirname(f))
                fname = os.path.basename(f)
                display_name = f"{parent_dir}/{fname}"

                file_item = QTreeWidgetItem([display_name])
                file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                file_item.setData(0, Qt.UserRole, f)
                file_item.setToolTip(0, f)

                if f in excluded:
                    file_item.setCheckState(0, Qt.Unchecked)
                else:
                    file_item.setCheckState(0, Qt.Checked)
                    all_unchecked = False

                grp_item.addChild(file_item)

            if all_unchecked:
                grp_item.setCheckState(0, Qt.Unchecked)

        self.tree_files.blockSignals(False)

    def on_tree_item_changed(self, item, column):
        path = item.data(0, Qt.UserRole)
        if path:
            is_checked = (item.checkState(0) == Qt.Checked)
            self.vm.set_file_excluded(path, not is_checked)

    def _on_ui_changed(self):
        self.vm.output_folder = self.txt_folder.text()
        self.vm.model_name = self.txt_name.text()
        self.vm.model_desc = self.txt_desc.text()
        self.vm.model_type = self.combo_model.currentText()
        self.vm.val_ratio = self.spin_ratio.value()
        self.vm.n_features = self.spin_bands.value()
        method_map = {
            "SPA": "spa",
            "Full Band": "full",
            "ANOVA-F": "anova_f",
            "SPA-LDA Fast": "spa_lda_fast",
            "SPA-LDA Greedy": "spa_lda_greedy",
            "LDA-coef": "lda_coef",
        }
        self.vm.band_selection_method = method_map.get(self.combo_band_method.currentText(), "spa")
        self.spin_bands.setEnabled(self.vm.band_selection_method != "full")
        self.vm.config_changed.emit()

    def _on_band_method_changed(self):  # AI가 수정함: Optimize 섹션 band method 변경 핸들러
        """Show/hide SPA-LDA Greedy warning based on current band method selection."""
        method_map = {
            "SPA": "spa", "Full Band": "full", "ANOVA-F": "anova_f",
            "SPA-LDA Fast": "spa_lda_fast", "SPA-LDA Greedy": "spa_lda_greedy", "LDA-coef": "lda_coef",
        }
        current = method_map.get(self.combo_band_method.currentText(), "spa")
        self.lbl_spa_greedy_warning.setVisible(current == "spa_lda_greedy")  # AI가 수정함: greedy 선택 시만 표시

    def connect_ui_events(self):
        self.txt_name.editingFinished.connect(self._on_ui_changed)
        self.txt_desc.editingFinished.connect(self._on_ui_changed)
        self.combo_model.currentIndexChanged.connect(self._on_ui_changed)
        self.spin_ratio.valueChanged.connect(self._on_ui_changed)
        self.spin_bands.valueChanged.connect(self._on_ui_changed)
        self.combo_band_method.currentIndexChanged.connect(self._on_ui_changed)
        self.combo_band_method.currentIndexChanged.connect(self._on_band_method_changed)
        self.list_experiment_band_methods.itemChanged.connect(self._on_experiment_selection_changed)
        self.list_experiment_models.itemChanged.connect(self._on_experiment_selection_changed)
        self.list_experiment_n_bands.itemChanged.connect(self._on_experiment_selection_changed)
        self.spin_experiment_gap_min.valueChanged.connect(self._on_experiment_gap_changed)
        self.spin_experiment_gap_max.valueChanged.connect(self._on_experiment_gap_changed)
        self.list_experiment_band_methods.itemChanged.connect(self._on_experiment_band_method_changed)

    def _restore_checked_items(self, list_widget, selected_values):
        selected_set = set(selected_values or [])
        list_widget.blockSignals(True)
        try:
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if item:
                    item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in selected_set else Qt.Unchecked)
        finally:
            list_widget.blockSignals(False)

    def _restore_checked_n_bands(self, selected_values):  # AI가 수정함: n_bands 체크 상태 복원
        """Restore checked state of list_experiment_n_bands from a list of int values."""
        if not hasattr(self, 'list_experiment_n_bands'):
            return
        selected_set = set(int(v) for v in (selected_values or []))
        self.list_experiment_n_bands.blockSignals(True)
        try:
            for index in range(self.list_experiment_n_bands.count()):
                item = self.list_experiment_n_bands.item(index)
                if item:
                    item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in selected_set else Qt.Unchecked)
        finally:
            self.list_experiment_n_bands.blockSignals(False)

    def _get_checked_values(self, list_widget):
        values = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item and item.checkState() == Qt.Checked:
                values.append(item.data(Qt.UserRole))
        return values

    def _get_checked_n_bands(self):  # AI가 수정함: 체크된 n_bands 값 목록 반환
        """Return list of checked n_bands integer values from list_experiment_n_bands."""
        if not hasattr(self, 'list_experiment_n_bands'):
            return []
        values = []
        for index in range(self.list_experiment_n_bands.count()):
            item = self.list_experiment_n_bands.item(index)
            if item and item.checkState() == Qt.Checked:
                values.append(item.data(Qt.UserRole))
        return values

    def _on_experiment_selection_changed(self, _item):
        self.vm.experiment_band_methods = self._get_checked_values(self.list_experiment_band_methods)
        self.vm.experiment_model_types = self._get_checked_values(self.list_experiment_models)
        self.vm.experiment_n_bands_list = self._get_checked_n_bands()
        self._update_experiment_summary()
        self.vm.config_changed.emit()

    def _on_experiment_gap_changed(self):  # AI가 수정함: gap spinbox 변경 핸들러
        """Sync gap range to VM and validate min <= max."""
        gap_min = self.spin_experiment_gap_min.value()
        gap_max = self.spin_experiment_gap_max.value()
        if gap_min > gap_max:  # AI가 수정함: min > max 방어
            self.spin_experiment_gap_max.setValue(gap_min)
            gap_max = gap_min
        self.vm.experiment_gap_min = gap_min
        self.vm.experiment_gap_max = gap_max
        self._update_experiment_summary()
        self.vm.config_changed.emit()

    def _on_experiment_band_method_changed(self, _item):  # AI가 수정함: band method 변경 시 SPA-LDA Greedy 경고 갱신
        """Show/hide Greedy warning and update gap widgets when band method selection changes."""
        checked_methods = self._get_checked_values(self.list_experiment_band_methods)
        self.lbl_experiment_spa_greedy_warning.setVisible("spa_lda_greedy" in checked_methods)
        self._update_gap_widgets_enabled()

    def _update_gap_widgets_enabled(self):  # AI가 수정함: prep_chain에 SimpleDeriv 없으면 gap 위젯 비활성화
        """Enable gap spinboxes only when SimpleDeriv is active in the prep chain."""
        has_simpledv = any(
            step.get('name') == 'SimpleDeriv'
            for step in self.vm.analysis_vm.prep_chain
        )
        self.spin_experiment_gap_min.setEnabled(has_simpledv)
        self.spin_experiment_gap_max.setEnabled(has_simpledv)
        self.lbl_experiment_gap_disabled.setVisible(not has_simpledv)

    def _set_form_signal_blocking(self, blocked):
        for widget in [self.txt_name, self.txt_desc, self.combo_model, self.spin_ratio, self.spin_bands, self.combo_band_method]:
            widget.blockSignals(blocked)

    def _update_experiment_summary(self):  # AI가 수정함: 4D 공식으로 업데이트
        band_count = len(self._get_checked_values(self.list_experiment_band_methods))
        model_count = len(self._get_checked_values(self.list_experiment_models))
        n_bands_count = len(self._get_checked_n_bands())
        has_simpledv = any(
            step.get('name') == 'SimpleDeriv'
            for step in getattr(self.vm, 'analysis_vm', None) and self.vm.analysis_vm.prep_chain or []
        )
        if has_simpledv:
            gap_min = self.spin_experiment_gap_min.value() if hasattr(self, 'spin_experiment_gap_min') else 1
            gap_max = self.spin_experiment_gap_max.value() if hasattr(self, 'spin_experiment_gap_max') else 20
            gap_count = max(0, gap_max - gap_min + 1)
        else:
            gap_count = 1  # AI가 수정함: SimpleDeriv 없으면 gap=1 (단일)
        trial_count = band_count * n_bands_count * gap_count * model_count
        self.lbl_experiment_summary.setText(
            f"Currently selected: {band_count} band methods × {n_bands_count} n_bands × {gap_count} gaps × {model_count} models = {trial_count} trial(s)."
        )

    def _toggle_experiment_section(self, checked):
        is_open = bool(checked)
        self.btn_toggle_experiment.setChecked(is_open)
        self.btn_toggle_experiment.setText("Hide advanced experiment options" if is_open else "Show advanced experiment options")
        self.experiment_content.setVisible(is_open)

    def append_log(self, msg):
        self.log_text.append(msg)

    def on_browse_click(self):
        current_path = self.txt_folder.text()
        initial_dir = current_path if current_path else "./output"
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            initial_dir,
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            full_path = folder_path.replace("/", "\\")
            self.txt_folder.setText(full_path)
            self.vm.output_folder = full_path
            self.vm.config_changed.emit()

    def on_export_click(self):  # AI가 수정함: n_bands_list, gap_range 포함하여 4D 실험 시작
        band_methods = self._get_checked_values(self.list_experiment_band_methods)
        model_types = self._get_checked_values(self.list_experiment_models)
        n_bands_list = self._get_checked_n_bands()
        if not band_methods:
            self.append_log("⚠️ Select at least one band method for paper summary export.")
            return
        if not model_types:
            self.append_log("⚠️ Select at least one model for paper summary export.")
            return
        if not n_bands_list:  # AI가 수정함: n_bands 미선택 시 현재 n_features로 fallback
            n_bands_list = [self.vm.n_features]
            self.append_log(f"ℹ️ No band count selected — using current n_features ({self.vm.n_features}).")
        gap_min = self.spin_experiment_gap_min.value()
        gap_max = self.spin_experiment_gap_max.value()
        if gap_min > gap_max:  # AI가 수정함: 유효성 검증
            self.append_log("⚠️ Gap min cannot exceed Gap max. Please fix the range.")
            return
        self.log_text.clear()
        self.set_buttons_enabled(False)
        self.vm.run_experiment_grid(band_methods, model_types, n_bands_list=n_bands_list, gap_min=gap_min, gap_max=gap_max)

    def on_open_folder_click(self):
        import subprocess
        folder = os.path.join(self.vm.output_folder, "experiments")
        if os.path.exists(folder):
            subprocess.Popen(f'explorer "{folder}"')
        else:
            self.append_log(f"⚠️ Folder not found: {folder}")
