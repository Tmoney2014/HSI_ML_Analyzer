import sys
import os
from PyQt5.QtWidgets import QApplication

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from views.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Optional: Set global style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
