from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

class ProjectWelcomeDialog(QDialog):
    """
    Startup dialog to force user to choose between New Project or Open Project.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to HSI Analyzer")
        self.setFixedSize(400, 250)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint) # No close button? Or allow close to exit app?
        # Better to allow close, but handle rejection in Main
        
        self.choice = None # 'new' or 'open'
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        lbl_title = QLabel("HSI Professional Analyzer")
        lbl_title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        lbl_title.setFont(font)
        layout.addWidget(lbl_title)
        
        lbl_sub = QLabel("Select an option to start:")
        lbl_sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_sub)
        
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(10)
        
        self.btn_new = QPushButton("📄   Create New Project")
        self.btn_new.setMinimumHeight(45)
        self.btn_new.setStyleSheet("font-size: 14px; font-weight: bold; text-align: left; padding-left: 20px;")
        self.btn_new.clicked.connect(self.on_new)
        
        self.btn_open = QPushButton("📂   Open Existing Project")
        self.btn_open.setMinimumHeight(45)
        self.btn_open.setStyleSheet("font-size: 14px; font-weight: bold; text-align: left; padding-left: 20px;")
        self.btn_open.clicked.connect(self.on_open)
        
        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_open)
        
        layout.addLayout(btn_layout)
        layout.addStretch()

    def on_new(self):
        self.choice = 'new'
        self.accept()

    def on_open(self):
        self.choice = 'open'
        self.accept()
