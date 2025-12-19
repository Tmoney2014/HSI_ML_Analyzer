import sys
import os
import json
import numpy as np
import traceback

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QLineEdit, QFileDialog, QSlider, QCheckBox, 
                             QGroupBox, QTextEdit, QMessageBox, QSplitter, QTabWidget, QListWidget,
                             QComboBox, QRadioButton, QButtonGroup, QProgressBar, QListWidgetItem,
                             QDialog, QSizePolicy, QProgressDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar # Removed default
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT 
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# Custom Toolbar Class
class CustomToolbar(NavigationToolbar2QT):
    # Only keep Save button
    toolitems = [
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    ]

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        # Make coordinate label bold
        if hasattr(self, 'locLabel'):
            font = self.locLabel.font()
            font.setBold(True)
            font.setPointSize(10)
            self.locLabel.setFont(font)

class ImageDetailWindow(QWidget):
    def __init__(self, path, parent=None):
        super().__init__()
        self.path = path
        self.main_app = parent
        self.setWindowTitle(f"Image Viewer: {os.path.basename(path)}")
        self.resize(1000, 800)
        
        self.layout = QVBoxLayout(self)
        
        # Top Controls
        hbox = QHBoxLayout()
        self.btn_toggle = QPushButton("Hide Graph ⏩")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False) # Default: Graph Visible (Button says "Hide")
        self.btn_toggle.clicked.connect(self.toggle_graph)
        hbox.addWidget(self.btn_toggle)
        
        self.btn_clear = QPushButton("Clear All Points 🗑️")
        self.btn_clear.clicked.connect(self.clear_all_points)
        hbox.addWidget(self.btn_clear)
        
        hbox.addStretch()
        self.layout.addLayout(hbox)
        
        # Canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        # Use Custom Toolbar
        self.toolbar = CustomToolbar(self.canvas, self)
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # Connect Events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # State
        self.pan_active = False
        self.pan_start = None
        self.selected_points = [] # List of {'id': int, 'x': int, 'y': int}
        self.point_counter = 1
        self.show_graph_panel = True
        
        # Color Palette (60 distinct colors)
        self.colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors

        # Load Data immediately
        self.cube = None
        self.load_data()
        self.update_view()

    def load_data(self):
        try:
            # Check if cached in main app first to save IO
            if hasattr(self.main_app, 'data_cache') and self.path in self.main_app.data_cache:
                self.cube, self.waves = self.main_app.data_cache[self.path]
            else:
                self.cube, self.waves = load_hsi_data(self.path)
            self.cube = np.nan_to_num(self.cube)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def toggle_graph(self):
        self.show_graph_panel = not self.show_graph_panel
        if self.show_graph_panel:
            self.btn_toggle.setText("Hide Graph ⏩")
        else:
            self.btn_toggle.setText("Show Graph ⏪")
        self.update_view()

    def update_view(self):
        if self.cube is None: return
        
        # PERSISTENCE: Save current limits
        saved_xlim = None
        saved_ylim = None
        if len(self.figure.axes) > 0:
            saved_xlim = self.figure.axes[0].get_xlim()
            saved_ylim = self.figure.axes[0].get_ylim()

        # Get Params from Main App
        try:
            is_ref = self.main_app.radio_ref.isChecked()
            if is_ref:
                threshold = self.main_app.slider_thresh.value() / 100.0
                data_viz = self.main_app.convert_to_reflectance(self.cube)
                self.processed_cube = data_viz 
            else:
                threshold = self.main_app.slider_thresh.value()
                data_viz = self.cube
                self.processed_cube = data_viz
            
            mask_input = self.main_app.get_mask_band_input()
            mask = create_background_mask(data_viz, threshold, mask_input)
            
            # Draw
            self.figure.clear()
            
            if self.show_graph_panel:
                gs = self.figure.add_gridspec(1, 2, width_ratios=[1, 1.2])
                self.ax_img = self.figure.add_subplot(gs[0, 0])
                self.ax_spec = self.figure.add_subplot(gs[0, 1])
                self.ax_spec.grid(True)
                self.ax_spec.set_title("Pixel Spectrum (Multi-Select)")
                self.ax_spec.set_xlabel("Wavelength / Band")
                self.ax_spec.set_ylabel("Intensity")
            else:
                self.ax_img = self.figure.add_axes([0, 0, 1, 1])
                self.ax_spec = None

            # --- Image Draw ---
            mid = data_viz.shape[2] // 2
            img_view = data_viz[:, :, mid].copy()
            if np.max(img_view) > 0: img_view /= np.max(img_view)
            
            img_rgb = np.dstack([img_view, img_view, img_view])
            img_rgb[~mask] = [0.0, 0.0, 0.0]
            
            self.ax_img.imshow(img_rgb)
            self.ax_img.set_aspect('equal', adjustable='datalim')
            self.ax_img.axis('off')
            self.ax_img.set_autoscale_on(False) # Prevent markers from zooming out view
            # HIDE COORDS on Image
            self.ax_img.format_coord = lambda x, y: "" 
            
            if self.show_graph_panel:
                self.ax_img.set_title(f"Image View")

            # --- Plot Selected Points ---
            for pt in self.selected_points:
                pid = pt['id']
                px, py = pt['x'], pt['y']
                
                # Pick color by ID (Stable)
                color = self.colors[pid % len(self.colors)]
                
                # 1. Marker on Image
                self.ax_img.plot(px, py, marker='+', markersize=12, markeredgewidth=2, color=color)
                # self.ax_img.text(px, py, str(pid), color=color, fontsize=10, fontweight='bold') # Removed text per user request
                
                # 2. Spectrum on Graph
                if self.ax_spec:
                    self.plot_spectrum_single(px, py, color, None)
                    # Custom Coordinate Format for Graph
                    self.ax_spec.format_coord = self.format_graph_coord

            # if self.ax_spec:
            #      self.ax_spec.legend(loc='upper right', fontsize='small') # Removed legend per user request

            # PERSISTENCE
            if saved_xlim is not None and saved_ylim is not None:
                self.ax_img.set_xlim(saved_xlim)
                self.ax_img.set_ylim(saved_ylim)
            
            if self.show_graph_panel:
                self.figure.tight_layout()
            
            rule_txt = mask_input if mask_input else 'Mean'
            self.setWindowTitle(f"Detail Viewer: {os.path.basename(self.path)} | Rules: {rule_txt} | Thresh: {threshold}")
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Viz Info: {e}")

    def format_graph_coord(self, x, y):
        # x is likely Wavelength, y is Intensity
        if hasattr(self, 'waves') and self.waves is not None and len(self.waves) > 0:
            # Find closest band index
            try:
                # Assuming visible range, find approximate index
                # This could be slow if done on every move? 150 items is fine.
                arr = np.array(self.waves)
                idx = (np.abs(arr - x)).argmin()
                return f"Band #{idx+1} ({x:.1f}nm), Int: {y:.0f}"
            except:
                return f"x={x:.1f}, y={y:.0f}"
        else:
            # No waves, x IS the band index
            return f"Band #{x:.0f}, Int: {y:.0f}"

    def plot_spectrum_single(self, x, y, color, label):
        if not hasattr(self, 'processed_cube'): return
        try:
            raw_spec = self.processed_cube[y, x, :].astype(float)
            
            # Apply Chain from Main App
            # Expand dims to (1, bands) for compatibility if functions expect 2D
            processed_spec = self.main_app.apply_preprocessing_chain(raw_spec[np.newaxis, :]).flatten()

            x_axis = self.waves if hasattr(self, 'waves') and self.waves else range(len(processed_spec))
            self.ax_spec.plot(x_axis, processed_spec, label=label, color=color, linewidth=1.5)
            
        except Exception as e:
            print(f"Plot Single Error: {e}")

    # --- Interaction Logic ---
    def on_scroll(self, event):
        ax = event.inaxes
        if ax != getattr(self, 'ax_img', None): return # Only zoom image
        
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None: return 

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.canvas.draw_idle()

    def clear_all_points(self):
        self.selected_points = []
        self.point_counter = 0
        self.update_view()

    def on_press(self, event):
        if event.inaxes != getattr(self, 'ax_img', None): return
        
        if event.button == 1: # Left Click: Select/Deselect
            # Fix 0.5px offset by rounding to nearest integer
            x, y = int(round(event.xdata)), int(round(event.ydata))
            
            # BOUNDARY CHECK
            if self.cube is not None:
                h, w, c = self.cube.shape
                if x < 0 or x >= w or y < 0 or y >= h:
                    return # Clicked outside
            
            # Check collisions
            collision_idx = -1
            radius = 1.0 # Exact match
            
            for idx, pt in enumerate(self.selected_points):
                px, py = pt['x'], pt['y']
                dist = np.sqrt((px - x)**2 + (py - y)**2)
                if dist < radius:
                    collision_idx = idx
                    break
            
            if collision_idx != -1:
                # Remove
                del self.selected_points[collision_idx]
            else:
                # Add with unique ID
                self.selected_points.append({'id': self.point_counter, 'x': x, 'y': y})
                self.point_counter += 1
                
            self.update_view() # Redraw
            
        elif event.button == 2: # Middle Click: Pan
            self.pan_active = True
            self.pan_start = (event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == 2:
            self.pan_active = False
            self.pan_start = None

    def on_motion(self, event):
        if self.pan_active and self.pan_start and event.inaxes == getattr(self, 'ax_img', None):
            ax = event.inaxes
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()

from utils.data_loader import load_hsi_data
from utils.preprocessing import create_background_mask, apply_mask, apply_snv, apply_savgol, apply_mean_centering, apply_l2_norm, apply_minmax_norm
from utils.band_selection import select_best_bands
from utils.model_trainer import train_model, export_model_for_csharp

class HSIAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Professional Analyzer v2.5")
        self.setGeometry(50, 50, 1600, 1000)
        
        # --- Data State ---
        self.normal_files = [] 
        self.defect_files = [] 
        self.white_ref_path = ""
        self.dark_ref_path = ""
        
        # --- Main UI ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { height: 40px; width: 200px; font-size: 14px; font-weight: bold; }")
        main_layout.addWidget(self.tabs)
        
        # Initialize Tabs
        self.init_tab_data()
        self.init_tab_analysis()
        self.init_tab_training()
        
        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Load Last Session
        self.load_session()

    def closeEvent(self, event):
        # Close all child windows
        if hasattr(self, 'img_windows'):
            for w in self.img_windows:
                w.close()
        self.save_session()
        event.accept()

    def save_session(self):
        cfg = {
            "normal_files": self.normal_files,
            "defect_files": self.defect_files,
            "white_ref": self.white_ref_path,
            "dark_ref": self.dark_ref_path,
            "threshold": self.txt_thresh.text(),
            "mask_band": self.txt_mask_band.text(),
            "sg_win": self.txt_sg_w.text(),
            "sg_poly": self.txt_sg_p.text()
        }
        try:
            with open("session_config.json", "w") as f:
                json.dump(cfg, f, indent=4)
            print("Session saved.")
        except Exception as e:
            print(f"Failed to save session: {e}")

    def load_session(self):
        if not os.path.exists("session_config.json"): return
        try:
            with open("session_config.json", "r") as f:
                cfg = json.load(f)
            
            self.normal_files = cfg.get("normal_files", [])
            self.defect_files = cfg.get("defect_files", [])
            self.white_ref_path = cfg.get("white_ref", "")
            self.dark_ref_path = cfg.get("dark_ref", "")
            
            # Restore UI
            self.txt_white.setText(self.white_ref_path)
            self.txt_dark.setText(self.dark_ref_path)
            
            if "threshold" in cfg: self.txt_thresh.setText(cfg["threshold"])
            if "mask_band" in cfg: self.txt_mask_band.setText(cfg["mask_band"])
            if "sg_win" in cfg: self.txt_sg_w.setText(cfg["sg_win"])
            if "sg_poly" in cfg: self.txt_sg_p.setText(cfg["sg_poly"])
            
            # Refresh Lists
            self.refresh_file_list()
            self.refresh_data_tabs()
            
            print(f"Session loaded: {len(self.normal_files)} Normal, {len(self.defect_files)} Defect files.")
            
        except Exception as e:
            print(f"Failed to load session: {e}")

    def refresh_data_tabs(self):
        self.list_normal.clear()
        for f in self.normal_files:
            item = f"{os.path.basename(f)}  [{os.path.dirname(f)}]"
            self.list_normal.addItem(item)
            
        self.list_defect.clear()
        for f in self.defect_files:
            item = f"{os.path.basename(f)}  [{os.path.dirname(f)}]"
            self.list_defect.addItem(item)
    def init_tab_data(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # --- Left: Normal Files ---
        grp_normal = QGroupBox("1. Normal Data (Batch)")
        vbox_n = QVBoxLayout()
        self.list_normal = QListWidget()
        self.list_normal.setSelectionMode(QListWidget.ExtendedSelection)
        vbox_n.addWidget(self.list_normal)
        
        hbox_btn_n = QHBoxLayout()
        btn_add_n = QPushButton("Add Folder/Files (+)")
        btn_add_n.clicked.connect(lambda: self.add_files(self.list_normal, self.normal_files))
        btn_del_n = QPushButton("Remove Selected (-)")
        btn_del_n.clicked.connect(lambda: self.remove_files(self.list_normal, self.normal_files))
        hbox_btn_n.addWidget(btn_add_n)
        hbox_btn_n.addWidget(btn_del_n)
        vbox_n.addLayout(hbox_btn_n)
        grp_normal.setLayout(vbox_n)
        layout.addWidget(grp_normal)
        
        # --- Center: Defect Files ---
        grp_defect = QGroupBox("2. Defect Data (Batch)")
        vbox_d = QVBoxLayout()
        self.list_defect = QListWidget()
        self.list_defect.setSelectionMode(QListWidget.ExtendedSelection)
        vbox_d.addWidget(self.list_defect)
        
        hbox_btn_d = QHBoxLayout()
        btn_add_d = QPushButton("Add Folder/Files (+)")
        btn_add_d.clicked.connect(lambda: self.add_files(self.list_defect, self.defect_files))
        btn_del_d = QPushButton("Remove Selected (-)")
        btn_del_d.clicked.connect(lambda: self.remove_files(self.list_defect, self.defect_files))
        hbox_btn_d.addWidget(btn_add_d)
        hbox_btn_d.addWidget(btn_del_d)
        vbox_d.addLayout(hbox_btn_d)
        grp_defect.setLayout(vbox_d)
        layout.addWidget(grp_defect)
        
        # --- Right: Reference & Mode ---
        grp_ref = QGroupBox("3. Reference & Mode")
        grp_ref.setFixedWidth(350)
        vbox_r = QVBoxLayout()
        
        # White Ref
        vbox_r.addWidget(QLabel("White Reference (.hdr):"))
        self.txt_white = QLineEdit()
        hbox_w = QHBoxLayout()
        hbox_w.addWidget(self.txt_white)
        btn_w = QPushButton("...")
        btn_w.setFixedWidth(30)
        btn_w.clicked.connect(lambda: self.browse_single_file(self.txt_white, "white"))
        hbox_w.addWidget(btn_w)
        vbox_r.addLayout(hbox_w)
        
        # Dark Ref
        vbox_r.addWidget(QLabel("Dark Reference (.hdr):"))
        self.txt_dark = QLineEdit()
        hbox_k = QHBoxLayout()
        hbox_k.addWidget(self.txt_dark)
        btn_k = QPushButton("...")
        btn_k.setFixedWidth(30)
        btn_k.clicked.connect(lambda: self.browse_single_file(self.txt_dark, "dark"))
        hbox_k.addWidget(btn_k)
        vbox_r.addLayout(hbox_k)
        
        vbox_r.addSpacing(20)
        
        # Mode Toggle
        grp_mode = QGroupBox("Processing Mode")
        vbox_m = QVBoxLayout()
        self.radio_raw = QRadioButton("Raw Data Mode (DN)")
        self.radio_ref = QRadioButton("Reflectance Mode")
        self.radio_raw.setChecked(True)
        self.radio_raw.toggled.connect(self.update_mode_ui)
        
        vbox_m.addWidget(self.radio_raw)
        vbox_m.addWidget(self.radio_ref)
        grp_mode.setLayout(vbox_m)
        vbox_r.addWidget(grp_mode)
        
        vbox_r.addStretch()
        grp_ref.setLayout(vbox_r)
        layout.addWidget(grp_ref)
        
        self.tabs.addTab(tab, "📂 Data Management")

    # =========================================================================
    # TAB 2: Analysis & Visualization
    # =========================================================================
    def init_tab_analysis(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # --- Left: Controls ---
        control_panel = QWidget()
        control_panel.setFixedWidth(400)
        vbox_c = QVBoxLayout(control_panel)
        
        # File Selector
        grp_sel = QGroupBox("Select Files")        # Data Lists
        split_lists = QSplitter(Qt.Vertical)
        
        self.list_viz = QListWidget()
        self.list_viz.itemChanged.connect(self.on_viz_item_changed)
        self.list_viz.itemDoubleClicked.connect(self.open_image_window) # New Connection
        split_lists.addWidget(QLabel("Select Files to Visualize (Double Click for Image View):"))
        split_lists.addWidget(self.list_viz)
        
        btn_refresh = QPushButton("Refresh File List")
        btn_refresh.clicked.connect(self.refresh_file_list)
        vbox_sel = QVBoxLayout()
        vbox_sel.addWidget(split_lists)
        vbox_sel.addWidget(btn_refresh)
        
        lbl_hint = QLabel("💡 Check box to see Mask & Spectrum.")
        lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        vbox_sel.addWidget(lbl_hint)
        
        grp_sel.setLayout(vbox_sel)
        vbox_c.addWidget(grp_sel)
        
        # Preprocessing Group
        grp_prep = QGroupBox("Preprocessing Parameters")
        vbox_p = QVBoxLayout()
        
        # --- Threshold & Mask Band ---
        vbox_p.addWidget(QLabel("<b>Background Masking Logic:</b>"))
        
        hbox_th = QHBoxLayout()
        hbox_th.addWidget(QLabel("Base Thresh:"))
        self.txt_thresh = QLineEdit("10") # Default
        self.txt_thresh.setFixedWidth(60)
        self.txt_thresh.returnPressed.connect(self.force_apply_thresh)
        hbox_th.addWidget(self.txt_thresh)
        vbox_p.addLayout(hbox_th)
        
        hbox_mask = QHBoxLayout()
        hbox_mask.addWidget(QLabel("Rules:"))
        self.txt_mask_band = QLineEdit("Mean")
        self.txt_mask_band.setToolTip("Examples:\n 'Mean' (Average)\n 'b50' (Band 50)\n 'b80>0.1 & b130<0.3' (AND)\n 'b80>0.1 | b130<0.3' (OR)")
        self.txt_mask_band.setPlaceholderText("e.g. b50 or b80>0.1")
        hbox_mask.addWidget(self.txt_mask_band)
        
        btn_apply = QPushButton("Apply")
        btn_apply.setFixedWidth(50)
        btn_apply.clicked.connect(self.force_apply_thresh)
        hbox_mask.addWidget(btn_apply)
        vbox_p.addLayout(hbox_mask)
        
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.valueChanged.connect(self.sync_slider_text)
        self.slider_thresh.sliderReleased.connect(self.request_preview_update)
        vbox_p.addWidget(self.slider_thresh)
        
        vbox_p.addSpacing(10)
        
        # Preprocessing List (Reorderable)
        vbox_p.addWidget(QLabel("<b>Preprocessing Pipeline (Drag to Reorder):</b>"))
        self.list_prep = QListWidget()
        self.list_prep.setDragDropMode(QListWidget.InternalMove)
        self.list_prep.setSelectionMode(QListWidget.SingleSelection)
        self.list_prep.itemChanged.connect(self.request_preview_update)
        self.list_prep.model().rowsMoved.connect(self.request_preview_update)
        
        # Defined Steps
        steps = [
            ("Savitzky-Golay Filter", "SG"),
            ("L2 Norm (Vector=1)", "L2"),
            ("Min-Max Norm (0-1)", "MinMax"),
            ("SNV (Standardize)", "SNV"),
            ("Mean Centering", "Center")
        ]
        
        for name, key in steps:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, key)
            self.list_prep.addItem(item)
            
        vbox_p.addWidget(self.list_prep)

        # SG Parameters
        hbox_sg = QHBoxLayout()
        hbox_sg.addWidget(QLabel("SG Win:"))
        self.txt_sg_w = QLineEdit("5")
        self.txt_sg_w.setFixedWidth(40)
        hbox_sg.addWidget(self.txt_sg_w)
        hbox_sg.addWidget(QLabel("Poly:"))
        self.txt_sg_p = QLineEdit("2")
        self.txt_sg_p.setFixedWidth(40)
        hbox_sg.addWidget(self.txt_sg_p)
        hbox_sg.addWidget(QLabel("Deriv:"))
        self.txt_sg_d = QLineEdit("0") # Derivative Order
        self.txt_sg_d.setFixedWidth(40)
        hbox_sg.addWidget(self.txt_sg_d)
        vbox_p.addLayout(hbox_sg)
        
        btn_upd = QPushButton("Update All Graphs")
        btn_upd.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 35px;")
        btn_upd.clicked.connect(self.request_preview_update)
        vbox_p.addWidget(btn_upd)
        
        btn_reset = QPushButton("Data Reload")
        btn_reset.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; height: 30px;")
        btn_reset.clicked.connect(self.reload_data_cache)
        vbox_p.addWidget(btn_reset)
        
        grp_prep.setLayout(vbox_p)
        vbox_c.addWidget(grp_prep)
        
        layout.addWidget(control_panel)
        
        # --- Right: Visualization ---
        viz_layout = QVBoxLayout()
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        
        # Toolbar
        self.toolbar = CustomToolbar(self.canvas, self)
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        # Connect Events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        # self.canvas.mpl_connect('button_press_event', self.on_canvas_click) # Removed double click logic

        # Wrapper widget
        viz_widget = QWidget()
        viz_widget.setLayout(viz_layout)
        layout.addWidget(viz_widget, stretch=1)
        
        self.tabs.addTab(tab, "📊 Analysis & Visualization")
        
        # Pan State
        self.pan_active = False
        self.pan_start = None

    # ... [Skipping methods] ...

    def apply_preprocessing_chain(self, data):
        # Helper to apply steps in ListWidget order
        processed = data.copy()
        for i in range(self.list_prep.count()):
            item = self.list_prep.item(i)
            if item.checkState() == Qt.Checked:
                key = item.data(Qt.UserRole)
                try:
                    if key == "SG":
                        w = int(self.txt_sg_w.text())
                        p = int(self.txt_sg_p.text())
                        try: d = int(self.txt_sg_d.text())
                        except: d = 0
                        processed = apply_savgol(processed, w, p, deriv=d)
                    elif key == "L2":
                        processed = apply_l2_norm(processed)
                    elif key == "MinMax":
                        processed = apply_minmax_norm(processed)
                    elif key == "SNV":
                        processed = apply_snv(processed)
                    elif key == "Center":
                        processed = apply_mean_centering(processed)
                    
                    # Safety: Handle NaNs immediately
                    processed = np.nan_to_num(processed)
                except: pass
        return processed

    def update_multiviz(self):
        # 1. Check Checked Items
        checked_items = []
        for i in range(self.list_viz.count()):
            item = self.list_viz.item(i)
            if item.checkState() == Qt.Checked:
                checked_items.append(item)
        
        # Sync to Detail Windows
        if hasattr(self, 'img_windows'):
            self.img_windows = [w for w in self.img_windows if w.isVisible()]
            for win in self.img_windows:
                win.update_view() # This might be heavy too, maybe skip if not focused?

        if not checked_items:
            self.figure.clear()
            self.figure.text(0.5, 0.5, "Check files on the left to see Spectrum", ha='center')
            self.canvas.draw()
            return

        # PROGRESS DIALOG
        progress = QProgressDialog("Processing Spectra...", "Cancel", 0, len(checked_items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0) # Show immediately
        progress.show()
        QApplication.processEvents() # Force UI update
        
        try:
            self.figure.clear()  
            # Spectrum Plot ONLY
            gs = gridspec.GridSpec(1, 1) 
            ax_spec = self.figure.add_subplot(gs[0, 0])
            ax_spec.format_coord = self.format_main_graph_coord
            
            # Reset current waves for formatter
            self.current_waves = None

            # Params
            is_ref = self.radio_ref.isChecked()
            if is_ref:
                threshold = self.slider_thresh.value() / 100.0
            else:
                threshold = self.slider_thresh.value()
            
            mask_input = self.get_mask_band_input()
            global_max_val = 100
            
            if not hasattr(self, 'data_cache'): self.data_cache = {}

            for idx, item in enumerate(checked_items):
                if progress.wasCanceled(): break
                progress.setValue(idx)
                QApplication.processEvents() # Keep UI responsive
                
                try:
                    path = item.data(Qt.UserRole)
                    fname = os.path.basename(path)
                    
                    # CACHE LOGIC: Check -> Load -> Save
                    if path in self.data_cache:
                        cube, waves = self.data_cache[path]
                    else:
                        cube, waves = load_hsi_data(path)
                        cube = np.nan_to_num(cube)
                        # Explicitly CACHE it now (Optimized)
                        self.data_cache[path] = (cube, waves)
                    
                    # Save waves for formatter (from first valid file)
                    if self.current_waves is None and waves is not None:
                         self.current_waves = np.array(waves)
                    
                    if not is_ref:
                        mx = np.max(cube)
                        if mx > global_max_val: global_max_val = mx
                        
                    if is_ref: data_viz = self.convert_to_reflectance(cube)
                    else: data_viz = cube
                        
                    mask = create_background_mask(data_viz, threshold, mask_input)
                    
                    # Apply Mask -> Flatten
                    valid_pixels = data_viz[mask]
                    
                    if valid_pixels.size == 0: continue
                    
                    # Calc Mean Spectrum
                    mean_spec = np.mean(valid_pixels, axis=0)
                    
                    # Apply Preprocessing Chain
                    mean_spec = self.apply_preprocessing_chain(mean_spec[np.newaxis, :]).flatten()

                    x_axis = waves if waves else range(len(mean_spec))
                    ax_spec.plot(x_axis, mean_spec, label=fname, alpha=0.7)

                except Exception as e:
                    print(f"Error {path}: {e}")
            
            progress.setValue(len(checked_items))
            
            if not is_ref and global_max_val > self.slider_thresh.maximum():
                 self.slider_thresh.setMaximum(int(global_max_val))

            ax_spec.legend(loc='upper right', fontsize='small')
            ax_spec.grid(True)
            ax_spec.set_title("Multi-Spectrum Comparison")
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.statusBar().showMessage(f"Visualization updated with {len(checked_items)} files.", 3000)
            
        finally:
            progress.close()

    def format_main_graph_coord(self, x, y):
        # x is likely Wavelength (if waves exist) or Band Index
        if hasattr(self, 'current_waves') and self.current_waves is not None:
             try:
                # Find closest band index
                idx = (np.abs(self.current_waves - x)).argmin()
                return f"Band #{idx+1} ({x:.1f}nm), Int: {y:.0f}"
             except: pass
             
        return f"x={x:.1f}, y={y:.2f}"


        # Update Dependent Image Windows
        if hasattr(self, 'img_windows'):
             # cleanup closed
            self.img_windows = [w for w in self.img_windows if w.isVisible()]
            for win in self.img_windows:
                win.update_view()
    
    # --- Pan Logic ---
    def on_press(self, event):
        if event.button == 2: # Middle Click
            self.pan_active = True
            self.pan_start = (event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == 2:
            self.pan_active = False
            self.pan_start = None

    def on_motion(self, event):
        if self.pan_active and self.pan_start and event.inaxes:
            ax = event.inaxes
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Shift limits (Pan direction is opposite to drag)
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            
            self.canvas.draw_idle()
        
    def open_image_window(self, item):
        path = item.data(Qt.UserRole)
        # Check if already open?
        # Use a list to keep references so they aren't GC'd
        if not hasattr(self, 'img_windows'): self.img_windows = []
        
        # Clean up closed windows from list
        self.img_windows = [w for w in self.img_windows if w.isVisible()]
        
        win = ImageDetailWindow(path, self)
        win.show()
        self.img_windows.append(win)
        
    def reload_data_cache(self):
        if hasattr(self, 'data_cache'):
            self.data_cache.clear()
        
        # Also clear cache in open popups?
        if hasattr(self, 'img_windows'):
             # cleanup closed
            self.img_windows = [w for w in self.img_windows if w.isVisible()]
            for win in self.img_windows:
                # Force reload in window
                win.load_data()
                win.update_view()

        self.status_bar.showMessage("Data cache cleared. Reloading...")
        self.request_preview_update()

    def reset_views(self):
        # Legacy or Zoom specific
        self.request_preview_update()

    def on_scroll(self, event):
        ax = event.inaxes
        if not ax: return
        
        # Only zoom images or spectrum 
        # Heuristic: spectrum plot has legend or grid, images behave differently
        # Let's just apply to all axes under cursor
        
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None: return

        # New Limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.canvas.draw_idle()

    def on_canvas_click(self, event):
        if event.dblclick and event.inaxes:
            # Check if clicked on an image subplot
            # Ideally store ax mapping, but for now just check if it's not the spectrum plot
            # Spectrum plot is usually the last one added or bottom one.
            # Simple heuristic: images have no axis labels usually
            
            try:
                # Find which image index
                ax = event.inaxes
                title = ax.get_title()
                if not title or title == "Multi-Spectrum Comparison": return
                
                # Check mapping data
                # We need to re-find the file path from title or store it better
                # Heuristic: search in loaded files
                target_path = None
                for f in self.normal_files + self.defect_files:
                    if os.path.basename(f) == title:
                        target_path = f
                        break
                        
                if target_path:
                    self.show_popup_viz(target_path)
            except: pass

    def show_popup_viz(self, path):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Detailed View: {os.path.basename(path)}")
        dlg.resize(800, 800)
        
        layout = QVBoxLayout(dlg)
        
        fig = Figure(figsize=(8, 12))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dlg)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        try:
            cube, _ = load_hsi_data(path)
            cube = np.nan_to_num(cube)
            
            is_ref = self.radio_ref.isChecked()
            if is_ref: 
                cube = self.convert_to_reflectance(cube)
                threshold = self.slider_thresh.value() / 100.0
            else:
                threshold = self.slider_thresh.value()
                
            mask_input = self.get_mask_band_input()
            mask = create_background_mask(cube, threshold, mask_input)
            
            ax = fig.add_subplot(111)
            mid = cube.shape[2] // 2
            img_view = cube[:, :, mid].copy()
            if np.max(img_view) > 0: img_view /= np.max(img_view)
            
            img_rgb = np.dstack([img_view, img_view, img_view])
            img_rgb[~mask] = [0.0, 0.0, 0.0] # Black BG
            
            ax.imshow(img_rgb)
            ax.set_title("Double-Click Zoom View")
            ax.axis('off')
            
            fig.tight_layout()
            canvas.draw()
            
        except Exception as e:
            layout.addWidget(QLabel(f"Error: {e}"))
            
        dlg.exec_()

    def log(self, msg):
        self.log_text.append(msg)
        self.status_bar.showMessage(msg)
        QApplication.processEvents()

    # =========================================================================
    # LOGIC
    # =========================================================================
    def add_files(self, list_widget, path_list):
        opts = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select HDR Files", "", "HDR Files (*.hdr);;All Files (*)", options=opts)
        if files:
            for f in files:
                if f not in path_list:
                    path_list.append(f)
                    item = f"{os.path.basename(f)}  [{os.path.dirname(f)}]"
                    list_widget.addItem(item)
            self.refresh_file_list()
            self.status_bar.showMessage(f"Added {len(files)} files.")

    def remove_files(self, list_widget, path_list):
        items = list_widget.selectedItems()
        if not items: return
        for item in items:
            idx = list_widget.row(item)
            if idx < len(path_list):
                del path_list[idx]
            list_widget.takeItem(idx)
        self.refresh_file_list()

    def browse_single_file(self, line_edit, type_ref):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Ref File", "", "HDR Files (*.hdr)")
        if fname:
            line_edit.setText(fname)
            if type_ref == "white": self.white_ref_path = fname
            if type_ref == "dark": self.dark_ref_path = fname

    def update_mode_ui(self):
        if self.radio_ref.isChecked():
            if not self.white_ref_path or not os.path.exists(self.white_ref_path):
                QMessageBox.warning(self, "Error", "Cannot switch to Reflectance Mode!\nPlease load a White Reference file first.")
                self.radio_raw.setChecked(True)
                return
                
            self.status_bar.showMessage("Mode: Reflectance (0.0 - 1.0)")
            self.slider_thresh.setRange(0, 100)
            self.slider_thresh.setValue(10) # 0.1 default
            self.txt_thresh.setText("0.1")
        else:
            self.status_bar.showMessage("Mode: Raw DN (0 - MAX)")
            self.slider_thresh.setRange(0, 4096)
            self.slider_thresh.setValue(100)
            self.txt_thresh.setText("100")
            
        self.request_preview_update()

    def refresh_file_list(self):
        self.list_viz.clear()
        
        for f in self.normal_files:
            item = QListWidgetItem(f"[Normal] {os.path.basename(f)}")
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, f)
            self.list_viz.addItem(item)
            
        for f in self.defect_files:
            item = QListWidgetItem(f"[Defect] {os.path.basename(f)}")
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, f)
            self.list_viz.addItem(item)

    def on_viz_item_changed(self, item):
        self.request_preview_update()

    def sync_slider_text(self):
        val = self.slider_thresh.value()
        if self.radio_ref.isChecked():
            # Slider 0-100 -> Float 0.0-1.0
            float_val = val / 100.0
            self.txt_thresh.setText(f"{float_val:.2f}")
        else:
            # Slider 0-MAX -> Int
            self.txt_thresh.setText(str(val))

    def force_apply_thresh(self):
        try:
            txt_val = float(self.txt_thresh.text())
            
            if self.radio_ref.isChecked():
                # Input 0.5 -> Slider 50
                slider_val = int(txt_val * 100)
                self.slider_thresh.setRange(0, 100) # Enforce
                self.slider_thresh.setValue(slider_val)
            else:
                # Input 4000 -> Slider 4000
                slider_val = int(txt_val)
                # Expand range if needed
                if slider_val > self.slider_thresh.maximum():
                    self.slider_thresh.setMaximum(slider_val + 1000)
                self.slider_thresh.setValue(slider_val)
                
            self.request_preview_update()
            
        except ValueError:
            pass

    def reload_data_cache(self):
        self.statusBar().showMessage("Data cache cleared. Reloading...")
        if hasattr(self, 'data_cache'):
            self.data_cache.clear()
        
        # Reloading Detail Windows will force-load
        if hasattr(self, 'img_windows'):
             # cleanup closed
            self.img_windows = [w for w in self.img_windows if w.isVisible()]
            for win in self.img_windows:
                # If we clear self.data_cache, requests will reload from disk
                pass

        self.request_preview_update()
        self.statusBar().showMessage("Ready", 3000)

    def request_preview_update(self):
        self.update_multiviz()

    def convert_to_reflectance(self, data_cube):
        if not os.path.exists(self.white_ref_path): return data_cube
        try:
            if not hasattr(self, 'cache_white') or self.cache_white_path != self.white_ref_path:
                w, _ = load_hsi_data(self.white_ref_path)
                self.cache_white = np.mean(w, axis=0)
                self.cache_white_path = self.white_ref_path
            d_vec = 0
            if os.path.exists(self.dark_ref_path):
                if not hasattr(self, 'cache_dark') or self.cache_dark_path != self.dark_ref_path:
                    d, _ = load_hsi_data(self.dark_ref_path)
                    self.cache_dark = np.mean(d, axis=0)
                    self.cache_dark_path = self.dark_ref_path
                d_vec = self.cache_dark
            
            numerator = data_cube - d_vec
            denominator = self.cache_white - d_vec
            denominator[denominator == 0] = 1e-6
            return np.clip(numerator / denominator, 0.0, 1.0)
        except: return data_cube

    def get_mask_band_input(self):
        # Returns parsed int or raw string info
        txt = self.txt_mask_band.text().strip()
        if txt.lower() == "mean" or txt == "":
            return None
        
        # Check for complex rules
        if any(op in txt for op in ['>', '<', ';', '&', '|']):
            return txt # Return string as-is for parser
            
        try:
            # Handle single band 'b50' or '50'
            clean_txt = txt.lower().replace("b", "")
            return int(clean_txt) 
        except:
            return None

    # =========================================================================
    # TAB 3: Training
    # =========================================================================
    def init_tab_training(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        grp_info = QGroupBox("Training Configuration")
        vbox_i = QVBoxLayout()
        label_info = QLabel("This will train a Linear SVM using ALL files listed in the Data Management tab.\nMake sure you have tuned the preprocessing parameters in the Analysis tab first.")
        vbox_i.addWidget(label_info)
        
        hbox_out = QHBoxLayout()
        hbox_out.addWidget(QLabel("Output Path:"))
        self.txt_output = QLineEdit("./output/model_config.json")
        hbox_out.addWidget(self.txt_output)
        vbox_i.addLayout(hbox_out)
        grp_info.setLayout(vbox_i)
        layout.addWidget(grp_info)
        
        btn_train = QPushButton("START BATCH TRAINING")
        btn_train.setStyleSheet("background-color: #2196F3; color: white; font-size: 16px; font-weight: bold; height: 50px;")
        btn_train.clicked.connect(self.run_batch_training)
        layout.addWidget(btn_train)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #333; color: #0f0; font-family: Consolas;")
        layout.addWidget(self.log_text)
        
        self.tabs.addTab(tab, "🎓 Training")

    # =========================================================================
    # Training
    # =========================================================================
    def run_batch_training(self):
        if not self.normal_files or not self.defect_files:
            QMessageBox.warning(self, "Error", "Add Normal and Defect files first!")
            return
            
        self.log("🚀 Starting Batch Training...")
        self.progress.setValue(0)
        
        try:
            is_ref = self.radio_ref.isChecked()
            if is_ref:
                threshold = float(self.txt_thresh.text())
            else:
                threshold = int(float(self.txt_thresh.text())) 

            mask_input = self.get_mask_band_input()

            prep_cfg = {
                "ApplySG": self.chk_sg.isChecked(),
                "SGWin": int(self.txt_sg_w.text()),
                "SGPoly": int(self.txt_sg_p.text()),
                "ApplyL2": self.chk_l2.isChecked(),
                "ApplyMinMax": self.chk_minmax.isChecked(),
                "ApplySNV": self.chk_snv.isChecked(),
                "ApplyCenter": self.chk_center.isChecked(),
                "Mode": "Reflectance" if is_ref else "Raw",
                "MaskRules": mask_input if mask_input is not None else "Mean" 
            }
            
            all_X = []
            all_y = []
            total_files = len(self.normal_files) + len(self.defect_files)
            processed = 0
            
            def process_file_list(files, label_val):
                nonlocal processed
                for f in files:
                    cube, _ = load_hsi_data(f)
                    cube = np.nan_to_num(cube)
                    
                    if prep_cfg["Mode"] == "Reflectance":
                         cube = self.convert_to_reflectance(cube)
                         
                    mask = create_background_mask(cube, threshold, mask_input) 
                    data = apply_mask(cube, mask)
                    
                    if prep_cfg["ApplySG"]: data = apply_savgol(data, prep_cfg["SGWin"], prep_cfg["SGPoly"])
                    if prep_cfg["ApplyL2"]: data = apply_l2_norm(data)
                    if prep_cfg["ApplyMinMax"]: data = apply_minmax_norm(data)
                    if prep_cfg["ApplySNV"]: data = apply_snv(data)
                    if prep_cfg["ApplyCenter"]: data = apply_mean_centering(data)
                    
                    if len(data) > 2000: data = data[np.random.choice(len(data), 2000, replace=False)]
                    
                    if len(data) > 0:
                        all_X.append(data)
                        all_y.append(np.full(len(data), label_val))
                    
                    processed += 1
                    self.progress.setValue(int(processed / total_files * 80))

            process_file_list(self.normal_files, 0)
            process_file_list(self.defect_files, 1)
            
            if len(all_X) == 0:
                raise ValueError("No valid data found after masking! Check Rules.")
                
            X_train = np.vstack(all_X)
            y_train = np.hstack(all_y)
            
            dummy_cube = X_train[:5000].reshape(-1, 1, X_train.shape[1])
            best_bands = select_best_bands(dummy_cube, n_bands=5)
            
            X_train_sub = X_train[:, best_bands]
            model = train_model(X_train_sub, y_train)
            
            out = self.txt_output.text()
            export_model_for_csharp(model, best_bands, out, prep_cfg)
            
            self.progress.setValue(100)
            QMessageBox.information(self, "Success", "Batch Training Finished!")
            
        except Exception as e:
            self.log(f"Training Error: {e}")
            print(traceback.format_exc())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HSIAnalysisApp()
    window.show()
    sys.exit(app.exec_())
