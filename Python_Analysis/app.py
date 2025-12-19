import sys
import os
import json
import numpy as np
import traceback

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QLineEdit, QFileDialog, QSlider, QCheckBox, 
                             QGroupBox, QTextEdit, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils.data_loader import load_hsi_data
from utils.preprocessing import create_background_mask, apply_mask, apply_snv, apply_savgol, apply_mean_centering, apply_l2_norm, apply_minmax_norm
from utils.band_selection import select_best_bands
from utils.model_trainer import train_model, export_model_for_csharp

class HSIAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Machine Learning Analyzer (PyQt5)")
        self.setGeometry(100, 100, 1400, 950)
        
        # Data storage
        self.cube_normal = None
        self.cube_defect = None
        self.wavelengths = None
        
        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # ----------------------
        # Left Panel (Controls)
        # ----------------------
        control_panel = QWidget()
        control_panel.setFixedWidth(400)
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)
        
        # Group 1: File Selection
        grp_files = QGroupBox("1. Data Selection (Raw DN)")
        layout_files = QVBoxLayout()
        
        self.path_normal = self.create_file_row("Normal (.hdr):", layout_files, r"C:\Users\user16g\Desktop\nonbr_br_fx50\0_2_non_br\capture\0_2_non_br.hdr")
        self.path_defect = self.create_file_row("Defect (.hdr):", layout_files, r"C:\Users\user16g\Desktop\nonbr_br_fx50\0_2_br_100_200\capture\0_2_br_0001.hdr")
        self.path_output = self.create_file_row("Output (.json):", layout_files, "./output/model_config.json")
        
        btn_load = QPushButton("Load Raw Data")
        btn_load.clicked.connect(self.load_data)
        btn_load.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        layout_files.addWidget(btn_load)
        
        grp_files.setLayout(layout_files)
        control_layout.addWidget(grp_files)
        
        # Group 2: Preprocessing
        grp_prep = QGroupBox("2. Preprocessing & Normalization")
        layout_prep = QVBoxLayout()
        
        # Threshold
        layout_thresh = QHBoxLayout()
        layout_thresh.addWidget(QLabel("BG Threshold (Raw):"))
        
        self.txt_thresh_val = QLineEdit("10")
        self.txt_thresh_val.setFixedWidth(60)
        self.txt_thresh_val.setAlignment(Qt.AlignRight)
        self.txt_thresh_val.returnPressed.connect(self.update_threshold_from_text)
        layout_thresh.addWidget(self.txt_thresh_val)
        
        btn_apply_thresh = QPushButton("Apply")
        btn_apply_thresh.setFixedWidth(50)
        btn_apply_thresh.clicked.connect(self.update_threshold_from_text)
        layout_thresh.addWidget(btn_apply_thresh)
        
        layout_prep.addLayout(layout_thresh)
        
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setMinimum(0)
        self.slider_thresh.setMaximum(100)
        self.slider_thresh.setValue(10)
        self.slider_thresh.valueChanged.connect(self.sync_slider_to_text)
        self.slider_thresh.sliderReleased.connect(self.update_preview)
        layout_prep.addWidget(self.slider_thresh)
        
        # Filters
        self.chk_sg = QCheckBox("Apply Savitzky-Golay Filter")
        layout_prep.addWidget(self.chk_sg)
        
        hbox_sg = QHBoxLayout()
        hbox_sg.addWidget(QLabel("Window:"))
        self.txt_sg_window = QLineEdit("5")
        self.txt_sg_window.setFixedWidth(40)
        hbox_sg.addWidget(self.txt_sg_window)
        hbox_sg.addWidget(QLabel("Poly:"))
        self.txt_sg_poly = QLineEdit("2")
        self.txt_sg_poly.setFixedWidth(40)
        hbox_sg.addWidget(self.txt_sg_poly)
        layout_prep.addLayout(hbox_sg)
        
        # Normalizations
        self.chk_snv = QCheckBox("Apply SNV (Standardize)")
        layout_prep.addWidget(self.chk_snv)
        
        self.chk_l2 = QCheckBox("Apply L2 Norm (Vector Length=1) [Recommend]")
        self.chk_l2.setStyleSheet("color: blue; font-weight: bold;")
        layout_prep.addWidget(self.chk_l2)
        
        self.chk_minmax = QCheckBox("Apply Min-Max Norm (0 to 1)")
        layout_prep.addWidget(self.chk_minmax)
        
        self.chk_center = QCheckBox("Apply Mean Centering")
        layout_prep.addWidget(self.chk_center)
        
        btn_preview = QPushButton("Update Preview")
        btn_preview.clicked.connect(self.update_preview)
        layout_prep.addWidget(btn_preview)
        
        grp_prep.setLayout(layout_prep)
        control_layout.addWidget(grp_prep)
        
        # Group 3: Training
        grp_train = QGroupBox("3. Training & Export")
        layout_train = QVBoxLayout()
        
        btn_train = QPushButton("Run Training Pipeline")
        btn_train.clicked.connect(self.run_training)
        btn_train.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        layout_train.addWidget(btn_train)
        
        grp_train.setLayout(layout_train)
        control_layout.addWidget(grp_train)
        
        # Log Box
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #f0f0f0; font-family: Consolas;")
        control_layout.addWidget(QLabel("Logs:"))
        control_layout.addWidget(self.log_text)
        
        # ----------------------
        # Right Panel (Viz)
        # ----------------------
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, stretch=1)
        
        self.ax_img = self.figure.add_subplot(211)
        self.ax_plot = self.figure.add_subplot(212)
        self.ax_img.set_title("Mask Preview (Load Data first)")
        self.ax_img.axis('off')
        self.ax_plot.set_title("Spectrum Preview (Raw DN)")

    def create_file_row(self, label_text, layout, default_val):
        lbl = QLabel(label_text)
        layout.addWidget(lbl)
        
        hbox = QHBoxLayout()
        line_edit = QLineEdit(default_val)
        hbox.addWidget(line_edit)
        
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(lambda: self.browse_file(line_edit))
        hbox.addWidget(btn_browse)
        
        layout.addLayout(hbox)
        return line_edit

    def browse_file(self, line_edit):
        fname, _ = QFileDialog.getOpenFileName(self, "Select File", "", "HDR Files (*.hdr);;All Files (*)")
        if fname:
            line_edit.setText(fname)
            
    def log(self, message):
        self.log_text.append(message)
        QApplication.processEvents()

    def sync_slider_to_text(self):
        val = self.slider_thresh.value()
        self.txt_thresh_val.setText(str(val))

    def update_threshold_from_text(self):
        try:
            val = int(self.txt_thresh_val.text())
            # Clamp value
            val = max(self.slider_thresh.minimum(), min(val, self.slider_thresh.maximum()))
            
            self.slider_thresh.setValue(val)
            self.txt_thresh_val.setText(str(val)) # Refresh text in case of clamp
            self.update_preview()
        except ValueError:
             # Revert to slider value if invalid input
            self.txt_thresh_val.setText(str(self.slider_thresh.value()))

    def load_data(self):
        self.log("📂 Loading Raw Data...")
        try:
            path_n = self.path_normal.text()
            path_d = self.path_defect.text()
            
            self.cube_normal, self.wavelengths = load_hsi_data(path_n)
            self.cube_normal = np.nan_to_num(self.cube_normal)
            
            self.cube_defect, _ = load_hsi_data(path_d)
            self.cube_defect = np.nan_to_num(self.cube_defect)
            
            self.log(f"✅ Normal Shape: {self.cube_normal.shape}")
            self.log(f"✅ Defect Shape: {self.cube_defect.shape}")
            
            # Update Slider Range based on Data Max
            max_val = np.max(self.cube_normal)
            self.slider_thresh.setMaximum(int(max_val))
            self.slider_thresh.setValue(int(max_val * 0.1)) # Default 10%
            self.log(f"ℹ️ Threshold Range Updated: 0 ~ {int(max_val)}")
            
            self.update_preview()
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))

    def update_preview(self):
        if self.cube_normal is None:
            return
            
        try:
            threshold = self.slider_thresh.value()
            
            # 1. Image Preview (Mask)
            self.cube_normal = np.array(self.cube_normal)
            if self.cube_normal.ndim != 3:
                raise ValueError(f"Normal Cube dimensionality error: {self.cube_normal.ndim}")
                
            mid_band = self.cube_normal.shape[2] // 2
            img_curr = self.cube_normal[:, :, mid_band]
            
            # Create mask
            mask = create_background_mask(self.cube_normal, threshold)
            
            # Visualization: Keep = Original (Gray), Remove = White
            # Normalize for display
            img_norm = img_curr.astype(float)
            if np.max(img_norm) > 0:
                img_norm /= np.max(img_norm)
            
            # Create RGB image
            img_rgb = np.dstack([img_norm, img_norm, img_norm])
            
            # Apply White to Background (~mask)
            img_rgb[~mask] = [1.0, 1.0, 1.0] # White
            
            self.ax_img.clear()
            self.ax_img.imshow(img_rgb)
            self.ax_img.set_title(f"Mask Visualization (White=Removed)\nThreshold: {threshold} (Raw DN)")
            self.ax_img.axis('off')
            
            # 2. Spectrum Preview
            data_n = apply_mask(self.cube_normal, mask)
            
            self.cube_defect = np.array(self.cube_defect)
            mask_d = create_background_mask(self.cube_defect, threshold)
            data_d = apply_mask(self.cube_defect, mask_d)
            
            if len(data_n) > 0 and len(data_d) > 0:
                # Subsample
                n_samp = 500
                if len(data_n) > n_samp: data_n = data_n[np.random.choice(len(data_n), n_samp, replace=False)]
                if len(data_d) > n_samp: data_d = data_d[np.random.choice(len(data_d), n_samp, replace=False)]
                
                # Apply Filters
                if self.chk_sg.isChecked():
                    w = int(self.txt_sg_window.text())
                    p = int(self.txt_sg_poly.text())
                    data_n = apply_savgol(data_n, w, p)
                    data_d = apply_savgol(data_d, w, p)
                
                # Normalizations (Order matters! Usually norm comes after smoothing)
                if self.chk_l2.isChecked():
                    data_n = apply_l2_norm(data_n)
                    data_d = apply_l2_norm(data_d)
                    
                if self.chk_minmax.isChecked():
                    data_n = apply_minmax_norm(data_n)
                    data_d = apply_minmax_norm(data_d)
                    
                if self.chk_snv.isChecked():
                    data_n = apply_snv(data_n)
                    data_d = apply_snv(data_d)
                    
                if self.chk_center.isChecked():
                    data_n = apply_mean_centering(data_n)
                    data_d = apply_mean_centering(data_d)
                
                mean_n = np.mean(data_n, axis=0)
                mean_d = np.mean(data_d, axis=0)
                
                # Use Wavelengths if available, else Band Index
                x_axis = self.wavelengths if (self.wavelengths is not None and len(self.wavelengths) == len(mean_n)) else range(len(mean_n))
                
                self.ax_plot.clear()
                self.ax_plot.plot(x_axis, mean_n, label='Normal', color='green')
                self.ax_plot.plot(x_axis, mean_d, label='Defect', color='red')
                self.ax_plot.legend()
                self.ax_plot.grid(True)
                
                xlabel = "Wavelength (nm)" if (self.wavelengths is not None and len(self.wavelengths) == len(mean_n)) else "Band Index"
                self.ax_plot.set_xlabel(xlabel)
                title_suffix = "(Normalized)" if (self.chk_l2.isChecked() or self.chk_snv.isChecked() or self.chk_minmax.isChecked()) else "(Raw DN)"
                self.ax_plot.set_title(f"Average Spectrum {title_suffix}")
            
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"⚠️ Preview Error: {e}")
            print(traceback.format_exc())

    def run_training(self):
        if self.cube_normal is None:
            QMessageBox.warning(self, "Warning", "Load data first!")
            return
            
        self.log("🚀 Starting Training...")
        try:
            threshold = self.slider_thresh.value()
            
            # 1. Masking
            self.log("   Applying background mask...")
            X_n = apply_mask(self.cube_normal, create_background_mask(self.cube_normal, threshold))
            X_d = apply_mask(self.cube_defect, create_background_mask(self.cube_defect, threshold))
            
            # 2. Preprocessing
            if self.chk_sg.isChecked():
                w = int(self.txt_sg_window.text())
                p = int(self.txt_sg_poly.text())
                self.log(f"   Filtering SG ({w}, {p})...")
                X_n = apply_savgol(X_n, w, p)
                X_d = apply_savgol(X_d, w, p)
                
            if self.chk_l2.isChecked():
                self.log("   Applying L2 Norm...")
                X_n = apply_l2_norm(X_n)
                X_d = apply_l2_norm(X_d)

            if self.chk_minmax.isChecked():
                self.log("   Applying Min-Max Norm...")
                X_n = apply_minmax_norm(X_n)
                X_d = apply_minmax_norm(X_d)
                
            if self.chk_snv.isChecked():
                self.log("   Applying SNV...")
                X_n = apply_snv(X_n)
                X_d = apply_snv(X_d)
                
            if self.chk_center.isChecked():
                self.log("   Applying Centering...")
                X_n = apply_mean_centering(X_n)
                X_d = apply_mean_centering(X_d)
                
            # 3. Band Selection
            self.log("🧠 Selecting Bands...")
            # Subsample for PCA
            n_pca = 3000
            idx_n = np.random.choice(len(X_n), min(n_pca, len(X_n)), replace=False)
            idx_d = np.random.choice(len(X_d), min(n_pca, len(X_d)), replace=False)
            X_pca = np.vstack([X_n[idx_n], X_d[idx_d]])
            
            dummy_cube = X_pca.reshape(-1, 1, X_pca.shape[1])
            selected_bands = select_best_bands(dummy_cube, n_bands=5)
            self.log(f"   Bands: {selected_bands}")
            
            # 4. Train
            self.log("🎓 Training SVM...")
            X_n_sub = X_n[:, selected_bands]
            X_d_sub = X_d[:, selected_bands]
            
            y_n = np.zeros(len(X_n_sub))
            y_d = np.ones(len(X_d_sub))
            
            X_train = np.vstack([X_n_sub, X_d_sub])
            y_train = np.hstack([y_n, y_d])
            
            if len(X_train) > 100000:
                idx = np.random.choice(len(X_train), 100000, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]
                
            model = train_model(X_train, y_train)
            
            # 5. Export
            out_path = self.path_output.text()
            prep_config = {
                "BackgroundThreshold": threshold,
                "ApplySG": self.chk_sg.isChecked(),
                "SGWindow": int(self.txt_sg_window.text()),
                "ApplyL2": self.chk_l2.isChecked(),
                "ApplyMinMax": self.chk_minmax.isChecked(),
                "ApplySNV": self.chk_snv.isChecked(),
                "ApplyCentering": self.chk_center.isChecked()
            }
            
            export_model_for_csharp(model, selected_bands, out_path, prep_config)
            
            self.log("✅ FINISHED! Config Saved.")
            QMessageBox.information(self, "Success", f"Model saved to:\n{out_path}")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Optional: Enable High DPI support
    # app.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    window = HSIAnalysisApp()
    window.show()
    sys.exit(app.exec_())
