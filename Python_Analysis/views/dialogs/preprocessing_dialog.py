"""
PreprocessingSettingsDialog for HSI ML Analyzer.
Extracted from tab_analysis.py for better code organization.
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLabel, 
                             QSpinBox, QComboBox,
                             QDialogButtonBox)


class PreprocessingSettingsDialog(QDialog):
    """Dialog for configuring preprocessing step parameters."""
    
    def __init__(self, method, params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Settings: {method}")
        self.key = method  # Store the key
        self.params = params or {}
        self.inputs = {}
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.inputs = {}
        
        if self.key == "SG":
            self.add_spin(form, "Window Size", "win", 5, 3, 51, 2)  # Odd numbers
            self.add_spin(form, "Poly Order", "poly", 2, 1, 5)
            self.add_spin(form, "Deriv Order", "deriv", 0, 0, 2)
            
        elif self.key == "SimpleDeriv":
            self.add_spin(form, "Gap Size", "gap", 1, 1, 50)
            
            # Order ComboBox
            cb_order = QComboBox()
            cb_order.addItems(["1st Derivative", "2nd Derivative"])
            current_order = self.params.get("order", 1)
            cb_order.setCurrentIndex(current_order - 1)
            form.addRow("Order:", cb_order)
            self.inputs["order"] = cb_order
            
        else:
            layout.addWidget(QLabel("No configurable parameters for this method."))
            
        layout.addLayout(form)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        
    def add_spin(self, layout, label, key, default, min_val, max_val, step=1):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(self.params.get(key, default))
        layout.addRow(f"{label}:", spin)
        self.inputs[key] = spin
        
    def get_params(self):
        """Extract parameters from dialog widgets."""
        new_params = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, QSpinBox):
                new_params[key] = widget.value()
            elif isinstance(widget, QComboBox):
                if key == "order":
                    new_params[key] = widget.currentIndex() + 1
        return new_params
