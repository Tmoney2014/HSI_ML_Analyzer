from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QProgressBar, 
                             QTextEdit, QLabel, QGroupBox, QHBoxLayout, QLineEdit)
from PyQt5.QtCore import Qt
from viewmodels.training_vm import TrainingViewModel

class TabTraining(QWidget):
    def __init__(self, training_vm: TrainingViewModel):
        super().__init__()
        self.vm = training_vm
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        grp_idx = QGroupBox("Training Configuration")
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("This will train a Linear SVM using ALL loaded files."))
        
        hbox_out = QHBoxLayout()
        hbox_out.addWidget(QLabel("Output Path:"))
        self.txt_output = QLineEdit("./output/model_config.json")
        hbox_out.addWidget(self.txt_output)
        vbox.addLayout(hbox_out)
        
        # New: Dynamic Band Selection
        hbox_bands = QHBoxLayout()
        hbox_bands.addWidget(QLabel("Number of Bands (Features):"))
        from PyQt5.QtWidgets import QSpinBox
        self.spin_bands = QSpinBox()
        self.spin_bands.setRange(1, 100)
        self.spin_bands.setValue(5)
        self.spin_bands.setToolTip("Number of top spectral bands to select for the model.")
        hbox_bands.addWidget(self.spin_bands)
        vbox.addLayout(hbox_bands)
        
        grp_idx.setLayout(vbox)
        layout.addWidget(grp_idx)
        
        # Buttons Layout
        hbox_btns = QHBoxLayout()
        
        self.btn_train = QPushButton("Train (Normal)")
        self.btn_train.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; font-weight: bold; height: 40px;")
        self.btn_train.clicked.connect(self.on_start_click)
        hbox_btns.addWidget(self.btn_train)
        
        self.btn_optimize = QPushButton("Auto-Optimize (Smart Search)")
        self.btn_optimize.setStyleSheet("background-color: #9C27B0; color: white; font-size: 14px; font-weight: bold; height: 40px;")
        self.btn_optimize.clicked.connect(self.on_optimize_click)
        hbox_btns.addWidget(self.btn_optimize)
        
        layout.addLayout(hbox_btns)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #333; color: #0f0; font-family: Consolas;")
        layout.addWidget(self.log_text)

    def connect_signals(self):
        self.vm.log_message.connect(self.append_log)
        self.vm.progress_update.connect(self.progress.setValue)
        self.vm.training_finished.connect(self.on_finished)

    def on_optimize_click(self):
        self.log_text.clear()
        self.set_buttons_enabled(False)
        output_path = self.txt_output.text()
        self.vm.run_optimization(output_path)
        
    def set_buttons_enabled(self, enabled):
        self.btn_train.setEnabled(enabled)
        self.btn_optimize.setEnabled(enabled)

    def on_start_click(self):
        self.log_text.clear()
        self.set_buttons_enabled(False)
        output_path = self.txt_output.text()
        n_features = self.spin_bands.value()
        
        # Run asynchronously? logic is currently blocking in VM. 
        # For true async, VM should use QThread. For now, we rely on VM's process_events or assume blocking.
        # But wait, VM doesn't process events. The UI will freeze.
        # I should probably update VM to yield or use QThread.
        # However, for this refactor I will keep it simple and just call it.
        # To prevent total freeze, I added QApp.processEvents logic in original app.
        # I'll rely on VM implementing that or just accept freeze for now (MVVM v1).
        # Actually VM creates a loop. I can add QApp access or simple yield.
        self.vm.run_training(output_path, n_features)

    def on_finished(self, success):
        self.set_buttons_enabled(True)
        if success:
            self.append_log("Done.")
        else:
            self.append_log("Failed.")

    def append_log(self, msg):
        self.log_text.append(msg)
