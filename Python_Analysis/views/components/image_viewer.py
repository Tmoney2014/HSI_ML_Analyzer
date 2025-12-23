import os
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from views.components.custom_toolbar import CustomToolbar
from services.data_loader import load_hsi_data
from models import processing

class ImageViewer(QWidget):
    def __init__(self, path, analysis_vm, parent=None):
        super().__init__()
        self.path = path
        self.vm = analysis_vm # AnalysisViewModel
        self.setWindowTitle(f"Image Viewer: {os.path.basename(path)}")
        self.resize(1000, 800)
        
        self.layout = QVBoxLayout(self)
        
        # Top Controls
        hbox = QHBoxLayout()
        self.btn_toggle = QPushButton("Hide Graph ⏩")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False)
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
        self.toolbar = CustomToolbar(self.canvas, self)
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # Events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # State
        self.pan_active = False
        self.pan_start = None
        self.selected_points = []
        self.point_counter = 1
        self.show_graph_panel = True
        
        self.colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
        
        # Data
        self.cube = None
        self.waves = None
        self.processed_cube = None
        
        self.load_data()
        self.update_view()

    def load_data(self):
        try:
            # Access cache via main_vm
            if self.path in self.vm.main_vm.data_cache:
                self.cube, self.waves = self.vm.main_vm.data_cache[self.path]
            else:
                self.cube, self.waves = load_hsi_data(self.path)
            self.cube = np.nan_to_num(self.cube)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def toggle_graph(self):
        self.show_graph_panel = not self.show_graph_panel
        self.btn_toggle.setText("Hide Graph ⏩" if self.show_graph_panel else "Show Graph ⏪")
        self.update_view()

    def update_view(self):
        if self.cube is None: return
        
        # Persistence
        saved_xlim = None
        saved_ylim = None
        if len(self.figure.axes) > 0:
            saved_xlim = self.figure.axes[0].get_xlim()
            saved_ylim = self.figure.axes[0].get_ylim()

        try:
            # 1. Logic (Sync with VM Params)
            is_ref = self.vm.use_ref
            threshold = self.vm.threshold
            
            if is_ref:
                data_viz = self.vm._convert_to_ref(self.cube) # Helper access
            else:
                data_viz = self.cube
            self.processed_cube = data_viz
            
            mask = processing.create_background_mask(data_viz, threshold, self.vm.mask_rules)
            
            # 2. Draw
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

            # Image
            mid = data_viz.shape[2] // 2
            img_view = data_viz[:, :, mid].copy()
            if np.max(img_view) > 0: img_view /= np.max(img_view)
            
            img_rgb = np.dstack([img_view, img_view, img_view])
            img_rgb[~mask] = [0.0, 0.0, 0.0]
            
            self.ax_img.imshow(img_rgb)
            self.ax_img.set_aspect('equal', adjustable='datalim')
            self.ax_img.axis('off')
            self.ax_img.set_autoscale_on(False)
            self.ax_img.format_coord = lambda x, y: "" 
            
            if self.show_graph_panel:
                self.ax_img.set_title(f"Image View")

            # Points
            for pt in self.selected_points:
                pid = pt['id']
                px, py = pt['x'], pt['y']
                color = self.colors[pid % len(self.colors)]
                
                self.ax_img.plot(px, py, marker='+', markersize=12, markeredgewidth=2, color=color)
                
                if self.ax_spec:
                    self.plot_spectrum_single(px, py, color, None)
                    self.ax_spec.format_coord = self.format_graph_coord

            # Restore View
            if saved_xlim is not None and saved_ylim is not None:
                self.ax_img.set_xlim(saved_xlim)
                self.ax_img.set_ylim(saved_ylim)
            
            if self.show_graph_panel:
                self.figure.tight_layout()
            
            self.setWindowTitle(f"Detail Viewer: {os.path.basename(self.path)}")
            self.canvas.draw()
            
        except Exception as e:
            print(f"Viz Info: {e}")

    def format_graph_coord(self, x, y):
        if self.waves and len(self.waves) > 0:
            try:
                arr = np.array(self.waves)
                idx = (np.abs(arr - x)).argmin()
                return f"Band #{idx+1} ({x:.1f}nm), Int: {y:.2f}"
            except: pass
        return f"x={x:.1f}, y={y:.2f}"

    def plot_spectrum_single(self, x, y, color, label):
        if self.processed_cube is None: return
        try:
            raw_spec = self.processed_cube[y, x, :].astype(float)
            processed_spec = raw_spec[np.newaxis, :] # (1, B)
            
            # Apply Chain locally using processing module
            for step in self.vm.prep_chain:
                name = step.get('name')
                p = step.get('params', {})
                if name == "SG":
                    processed_spec = processing.apply_savgol(processed_spec, p.get('win'), p.get('poly'), p.get('deriv', 0))
                elif name == "SimpleDeriv":
                    processed_spec = processing.apply_simple_derivative(processed_spec, gap=p.get('gap', 5), order=p.get('order', 1), apply_ratio=p.get('ratio', False), ndi_threshold=p.get('ndi_threshold', 1e-4))
                elif name == "SNV":
                    processed_spec = processing.apply_snv(processed_spec)
                elif name == "3PointDepth":
                    processed_spec = processing.apply_rolling_3point_depth(processed_spec, gap=p.get('gap', 5))
                elif name == "L2":
                    processed_spec = processing.apply_l2_norm(processed_spec)
                elif name == "MinSub":
                    processed_spec = processing.apply_min_subtraction(processed_spec)
                elif name == "MinMax":
                    processed_spec = processing.apply_minmax_norm(processed_spec)
                elif name == "Center":
                    processed_spec = processing.apply_mean_centering(processed_spec)
                processed_spec = np.nan_to_num(processed_spec)

            spec_1d = processed_spec.flatten()
            x_axis = self.waves if self.waves else range(len(spec_1d))
            
            # Fix: Gap Difference reduces band count, causing shape mismatch
            if len(x_axis) > len(spec_1d):
                x_axis = x_axis[:len(spec_1d)]
                
            self.ax_spec.plot(x_axis, spec_1d, label=label, color=color, linewidth=1.5)
            
        except Exception as e:
            # print(f"DEBUG: Processing Module: {processing.__file__}")
            print(f"Plot Single Error: {e}")

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None: return
        
        # Allow zoom on both Image and Graph
        if ax != getattr(self, 'ax_img', None) and ax != getattr(self, 'ax_spec', None):
            return
        
        base_scale = 1.2
        scale_factor = 1/base_scale if event.button == 'up' else base_scale
        
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
        ax = event.inaxes
        
        # Left Click: Point Selection (Only on Image)
        if event.button == 1 and ax == getattr(self, 'ax_img', None):
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if self.cube is not None:
                h, w, c = self.cube.shape
                if x < 0 or x >= w or y < 0 or y >= h: return
            
            # Collision
            collision_idx = -1
            for idx, pt in enumerate(self.selected_points):
                if np.sqrt((pt['x'] - x)**2 + (pt['y'] - y)**2) < 1.0:
                    collision_idx = idx
                    break
            
            if collision_idx != -1: del self.selected_points[collision_idx]
            else:
                self.selected_points.append({'id': self.point_counter, 'x': x, 'y': y})
                self.point_counter += 1
            self.update_view()
            
        # Middle Click: Pan (Both Image and Graph)
        elif event.button == 2:
            if ax == getattr(self, 'ax_img', None) or ax == getattr(self, 'ax_spec', None):
                self.pan_active = True
                self.pan_start = (event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == 2:
            self.pan_active = False
            self.pan_start = None

    def on_motion(self, event):
        if self.pan_active and self.pan_start and event.inaxes:
            ax = event.inaxes
            # Allow pan on both Image and Graph
            if ax != getattr(self, 'ax_img', None) and ax != getattr(self, 'ax_spec', None):
                return
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()
