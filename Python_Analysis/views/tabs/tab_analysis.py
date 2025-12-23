from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox, 
                             QListWidget, QPushButton, QLabel, QLineEdit, QSlider, QListWidgetItem, QProgressDialog, QApplication, QMessageBox, QDialog, QFormLayout, QSpinBox, QComboBox, QDialogButtonBox, QCheckBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import os
import sys
import numpy as np

from viewmodels.main_vm import MainViewModel
from viewmodels.analysis_vm import AnalysisViewModel
from views.components.custom_toolbar import CustomToolbar
from views.components.image_viewer import ImageViewer

class PreprocessingSettingsDialog(QDialog):
    def __init__(self, method, params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Settings: {method}")
        self.key = method # Store the key
        self.params = params or {}
        self.inputs = {}
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.inputs = {}
        
        if self.key == "SG":
            self.add_spin(form, "Window Size", "win", 5, 3, 51, 2) # Odd numbers
            self.add_spin(form, "Poly Order", "poly", 2, 1, 5)
            self.add_spin(form, "Deriv Order", "deriv", 0, 0, 2)
            
        elif self.key == "SimpleDeriv":
            self.add_spin(form, "Gap Size", "gap", 5, 1, 50)
            
            # Order ComboBox
            cb_order = QComboBox()
            cb_order.addItems(["1st Derivative", "2nd Derivative"])
            current_order = self.params.get("order", 1)
            cb_order.setCurrentIndex(current_order - 1)
            form.addRow("Order:", cb_order)
            self.inputs["order"] = cb_order
            
            # Application Ratio (NDI) Checkbox
            from PyQt5.QtWidgets import QCheckBox
            cb_ratio = QCheckBox("Use Normalized Ratio (NDI)")
            cb_ratio.setToolTip("Formula: (A-B) / (A+B)\nGood for lighting invariant analysis.")
            cb_ratio.setChecked(self.params.get("ratio", False))
            form.addRow("", cb_ratio)
            self.inputs["ratio"] = cb_ratio
            
            # NDI Threshold
            txt_thresh = QLineEdit()
            txt_thresh.setPlaceholderText("e.g. 1e-4")
            txt_thresh.setText(str(self.params.get("ndi_threshold", 1e-4)))
            form.addRow("NDI Threshold:", txt_thresh)
            self.inputs["ndi_threshold"] = txt_thresh
            
        elif self.key == "3PointDepth":
            self.add_spin(form, "Gap Size (Shoulder Distance)", "gap", 5, 1, 50)
            
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
        new_params = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, QSpinBox):
                new_params[key] = widget.value()
            elif isinstance(widget, QComboBox):
                if key == "order":
                    new_params[key] = widget.currentIndex() + 1
            elif isinstance(widget, QCheckBox): # Handle Checkbox
                new_params[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                if key == "ndi_threshold":
                    try:
                        new_params[key] = float(widget.text())
                    except:
                        new_params[key] = 1e-4
        return new_params

class TabAnalysis(QWidget):
    def __init__(self, main_vm: MainViewModel, analysis_vm: AnalysisViewModel):
        super().__init__()
        self.main_vm = main_vm
        self.analysis_vm = analysis_vm
        self.img_windows = []
        
        # Initialize UI widgets to None for safety
        self.list_prep = None
        self.slider_thresh = None
        self.list_viz = None
        self.list_viz = None
        self.txt_thresh = None

        # Flag to prevent signal loops during manual update
        self.updating_from_ui = False
        
        print("[TabAnalysis] __init__ started")
        self.init_ui()
        print("[TabAnalysis] init_ui completed")
        
        # Connect Signals
        self.main_vm.files_changed.connect(self.refresh_file_list)
        self.main_vm.mode_changed.connect(self.on_mode_changed)
        self.analysis_vm.error_occurred.connect(self.on_vm_error)
        self.analysis_vm.model_updated.connect(self.on_model_updated)

    def on_vm_error(self, msg):
        QMessageBox.critical(self, "Processing Error", f"An error occurred in the analysis engine:\n{msg}")

    def on_model_updated(self):
        """
        Sync UI with VM state (received from Optimization or other external source).
        """
        if self.updating_from_ui:
            return

        try:
            self.list_prep.blockSignals(True)
            if self.txt_thresh: self.txt_thresh.blockSignals(True)
            
            # 1. Update Threshold
            if self.txt_thresh:
                self.txt_thresh.setText(str(self.analysis_vm.threshold))
                
            # 2. Update List Params & Checked State
            # Map current VM chain steps by name -> params
            vm_chain_map = {step['name']: step['params'] for step in self.analysis_vm.prep_chain}
            
            for i in range(self.list_prep.count()):
                item = self.list_prep.item(i)
                key = item.data(Qt.UserRole)
                
                if key in vm_chain_map:
                    # Optimized Step: Update params and ensure checked
                    item.setCheckState(Qt.Checked)
                    # Only update if different to avoid signal loops
                    vm_p = vm_chain_map[key]
                    current_p = item.data(Qt.UserRole + 1)
                    
                    if current_p != vm_p:
                        item.setData(Qt.UserRole + 1, vm_p)
                else:
                    # Item not in chain
                    item.setCheckState(Qt.Unchecked)
                    
            # 3. Refresh Plot
            self.update_viz()
            
        finally:
            self.list_prep.blockSignals(False)
            if self.txt_thresh: self.txt_thresh.blockSignals(False)

    # ... (skipping methods)

    def init_ui(self):
        print("[TabAnalysis] init_ui started")
        layout = QHBoxLayout(self)
        
        # --- Left Panel ---
        control_panel = QWidget()
        control_panel.setFixedWidth(400)
        vbox_c = QVBoxLayout(control_panel)
        
        # File Selector
        grp_sel = QGroupBox("Select Files")
        split_lists = QSplitter(Qt.Vertical)
        self.list_viz = QListWidget()
        self.list_viz.itemDoubleClicked.connect(self.open_image_window)
        self.list_viz.itemChanged.connect(self.update_viz)
        split_lists.addWidget(QLabel("Select Files (Double Click for Image View):"))
        split_lists.addWidget(self.list_viz)
        
        btn_refresh = QPushButton("Refresh File List")
        btn_refresh.clicked.connect(self.refresh_file_list)
        vbox_sel = QVBoxLayout()
        vbox_sel.addWidget(split_lists)
        vbox_sel.addWidget(btn_refresh)
        vbox_sel.addWidget(QLabel("💡 Check box to see Mask & Spectrum."))
        grp_sel.setLayout(vbox_sel)
        vbox_c.addWidget(grp_sel)
        
        vbox_c.addWidget(grp_sel)
        
        # Preprocessing
        grp_prep = QGroupBox("Preprocessing Parameters")
        vbox_p = QVBoxLayout()
        
        # --- Threshold & Mask Band ---
        vbox_p.addWidget(QLabel("<b>Background Removal (Intensity Cutoff):</b>"))
        
        hbox_th = QHBoxLayout()
        hbox_th.addWidget(QLabel("Min Intensity (Cutoff):"))
        self.txt_thresh = QLineEdit("10")
        self.txt_thresh.setFixedWidth(60)
        self.txt_thresh.returnPressed.connect(self.update_params)
        hbox_th.addWidget(self.txt_thresh)
        vbox_p.addLayout(hbox_th)
        
        # Exclude Bands
        hbox_ex = QHBoxLayout()
        hbox_ex.addWidget(QLabel("Exclude Bands:"))
        self.txt_exclude = QLineEdit()
        self.txt_exclude.setPlaceholderText("e.g. 1-5, 92")
        self.txt_exclude.editingFinished.connect(self.update_params) # Update on lose focus/enter
        hbox_ex.addWidget(self.txt_exclude)
        vbox_p.addLayout(hbox_ex)
        
        hbox_mask = QHBoxLayout()
        hbox_mask.addWidget(QLabel("Rules:"))
        self.txt_mask_band = QLineEdit("Mean")
        self.txt_mask_band.setPlaceholderText("e.g. b50 or b80>0.1")
        hbox_mask.addWidget(self.txt_mask_band)
        btn_apply = QPushButton("Apply")
        btn_apply.setFixedWidth(50)
        btn_apply.clicked.connect(self.update_params)
        hbox_mask.addWidget(btn_apply)
        vbox_p.addLayout(hbox_mask)
        
        vbox_p.addWidget(QLabel("⬇ Drag slider to exclude background pixels:"))
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(10)
        # Real-time visual feedback
        self.slider_thresh.valueChanged.connect(self.on_slider_value_changed)
        # Heavy update on release
        self.slider_thresh.sliderReleased.connect(self.on_slider_release)
        vbox_p.addWidget(self.slider_thresh)
        
        vbox_p.addSpacing(10)
        vbox_p.addWidget(QLabel("<b>Preprocessing Pipeline (Drag to Reorder):</b>"))
        vbox_p.addWidget(QLabel("💡 Drag to reorder steps. Double-click to configure."))
        self.list_prep = QListWidget()
        self.list_prep.setDragDropMode(QListWidget.InternalMove)
        self.list_prep.itemChanged.connect(self.update_params)
        self.list_prep.model().rowsMoved.connect(self.update_params)
        self.list_prep.itemDoubleClicked.connect(self.open_prep_settings)
        
        steps = [
            ("Baseline Correction (Min Sub)", "MinSub"),
            ("Savitzky-Golay (Smoothing/Deriv)", "SG"),
            ("Simple Derivative (Gap Diff)", "SimpleDeriv"),
            ("L2 Normalization", "L2"),
            ("Min-Max Normalization", "MinMax"),
            ("Standard Normal Variate (SNV)", "SNV"),
            ("Mean Centering", "Center"),
            ("Rolling 3-Point Depth", "3PointDepth")
        ]
        for name, key in steps:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, key)
            item.setData(Qt.UserRole + 1, {}) # Store params dict here
            
            # Set default params
            if key == "SG":
                item.setData(Qt.UserRole + 1, {"win": 5, "poly": 2, "deriv": 0})
            elif key == "SimpleDeriv":
                item.setData(Qt.UserRole + 1, {"gap": 5, "order": 1, "ratio": False, "ndi_threshold": 1e-4})
            elif key == "3PointDepth":
                item.setData(Qt.UserRole + 1, {"gap": 5})
            elif key == "MinSub":
                item.setData(Qt.UserRole + 1, {})
                
            self.list_prep.addItem(item)
        vbox_p.addWidget(self.list_prep)
        
        btn_upd = QPushButton("Update All Graphs")
        btn_upd.clicked.connect(self.update_params)
        vbox_p.addWidget(btn_upd)
        
        btn_reload = QPushButton("Data Reload (Clear Cache)")
        btn_reload.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        btn_reload.clicked.connect(self.reload_data_cache)
        vbox_p.addWidget(btn_reload)
        
        grp_prep.setLayout(vbox_p)
        vbox_c.addWidget(grp_prep)
        
        layout.addWidget(control_panel)
        
        # --- Right Panel ---
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CustomToolbar(self.canvas, self)
        
        viz_layout = QVBoxLayout()
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        viz_widget = QWidget()
        viz_widget.setLayout(viz_layout)
        layout.addWidget(viz_widget, stretch=1)
        
        # Connect Events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Pan State
        self.pan_active = False
        self.pan_start = None

    def on_mode_changed(self, is_ref):
        if is_ref:
             self.slider_thresh.setRange(0, 1000)
             self.slider_thresh.setValue(10) # 0.01
             self.txt_thresh.setText("0.01")
        else:
             self.slider_thresh.setRange(0, 65535) # Safe max for both 12-bit and 16-bit
             self.slider_thresh.setValue(100)
             self.txt_thresh.setText("100")
        self.update_params()

    def on_slider_value_changed(self, val):
        """Update text only for visual feedback during drag"""
        try:
            if self.analysis_vm.use_ref:
                # 0-1000 -> 0.0-1.0
                self.txt_thresh.setText(f"{val/1000.0:.3f}")
            else:
                self.txt_thresh.setText(str(val))
        except Exception:
            pass # Ignore errors during rapid drag

    def on_slider_release(self):
        """Update model/viz and Autosave on release"""
        try:
            val = self.slider_thresh.value()
            self.update_params()
        except Exception:
            pass

    def open_prep_settings(self, item):
        key = item.data(Qt.UserRole)
        params = item.data(Qt.UserRole + 1)
        
        
        # Non-configurable methods: Ignore Double Click
        if key in ["MinSub", "L2", "MinMax", "SNV", "Center"]:
            return
            
        dlg = PreprocessingSettingsDialog(key, params, self)
        if dlg.exec_() == QDialog.Accepted:
            new_params = dlg.get_params()
            item.setData(Qt.UserRole + 1, new_params)
            
            # Construct human-readable label appendage
            base_name = item.text().split(" [")[0] # Removing existing params if any
            if key == "SG":
                lbl = f"{base_name} [w={new_params['win']}, p={new_params['poly']}, d={new_params['deriv']}]"
            elif key == "SimpleDeriv":
                lbl = f"{base_name} [Gap={new_params['gap']}, Ord={new_params['order']}]"
                if new_params.get("ratio", False):
                    lbl += f" (NDI, Th={new_params.get('ndi_threshold', 1e-4)})"
            elif key == "3PointDepth":
                 lbl = f"{base_name} [Gap={new_params['gap']}]"
            else:
                lbl = base_name
                
            item.setText(lbl)
            item.setText(lbl)
            # self.update_params() # Handled by itemChanged signal (setText triggers it)
        
    def refresh_file_list(self):
        try:
            self.list_viz.clear()
            files = self.main_vm.get_all_files()
            for f in files:
                item = QListWidgetItem(f"{os.path.basename(f)}  [{os.path.dirname(f)}]")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, f)
                self.list_viz.addItem(item)
        except RuntimeError:
            # Widget deleted during shutdown - ignore
            return
        except Exception as e:
            print(f"Error in refresh_file_list: {e}")
            import traceback
            traceback.print_exc()

    def on_mode_changed(self, is_ref):
        if is_ref:
             self.slider_thresh.setRange(0, 1000)
             self.slider_thresh.setValue(10) # 0.01
             self.txt_thresh.setText("0.01")
        else:
             self.slider_thresh.setRange(0, 65535) # Safe max for both 12-bit and 16-bit
             self.slider_thresh.setValue(100)
             self.txt_thresh.setText("100")
        self.update_params()

    def on_slider_release(self):
        val = self.slider_thresh.value()
        if self.analysis_vm.use_ref:
            # 0-1000 -> 0.0-1.0
            self.txt_thresh.setText(f"{val/1000.0:.3f}")
        else:
            self.txt_thresh.setText(str(val))
        self.update_params()

    def restore_ui(self):
        """Restore UI State from ViewModel"""
        # Safety Check: If UI is not fully initialized, do not attempt to restore
        if self.list_prep is None or self.slider_thresh is None:
            return

        try:
            # Threshold
            if self.analysis_vm.use_ref:
                self.slider_thresh.setRange(0, 1000)
                # Ensure valid float
                val = float(self.analysis_vm.threshold)
                self.slider_thresh.setValue(int(val * 1000))
                self.txt_thresh.setText(f"{val:.3f}")
            else:
                self.slider_thresh.setRange(0, 65535)
                val = int(self.analysis_vm.threshold)
                self.slider_thresh.setValue(val)
                self.txt_thresh.setText(str(val))
                
            # Mask Rules
            if self.analysis_vm.mask_rules:
                self.txt_mask_band.setText(str(self.analysis_vm.mask_rules))
                
            # Exclude Bands
            if self.analysis_vm.exclude_bands_str:
                self.txt_exclude.setText(self.analysis_vm.exclude_bands_str)
            
            # Preprocessing Chain
            # First uncheck all
            for i in range(self.list_prep.count()):
                self.list_prep.item(i).setCheckState(Qt.Unchecked)
            
            # Recheck and restore params
            for step in self.analysis_vm.prep_chain:
                name = step.get('name')
                p = step.get('params', {})
                
                # Check Item and Set Params
                for i in range(self.list_prep.count()):
                    item = self.list_prep.item(i)
                    if item.data(Qt.UserRole) == name:
                        item.setCheckState(Qt.Checked)
                        item.setData(Qt.UserRole + 1, p) # Restore Params
                        
                        # Update Label
                        base_name = item.text().split(" [")[0]
                        lbl = base_name
                        if name == "SG":
                            lbl = f"{base_name} [w={p.get('win',5)}, p={p.get('poly',2)}, d={p.get('deriv',0)}]"
                        elif name == "SimpleDeriv":
                            lbl = f"{base_name} [Gap={p.get('gap',5)}, Ord={p.get('order',1)}]"
                            if p.get("ratio", False): lbl += f" (NDI, Th={p.get('ndi_threshold', 1e-4)})"
                        elif name == "3PointDepth":
                            lbl = f"{base_name} [Gap={p.get('gap', 5)}]"
                        item.setText(lbl)
                        break
                        
            # Trigger Viz Update if any files checked
            self.update_viz()
            
        except Exception as e:
            print(f"Restore UI Error: {e}")

    def update_params(self, item=None):
        if self.list_prep is None or self.slider_thresh is None:
            return
            
        try:
            self.updating_from_ui = True
            
            # 1. Text -> VM
            val_float = float(self.txt_thresh.text())
            # Use setter to trigger params_changed signal
            self.analysis_vm.set_threshold(val_float)
            
            # 2. Text -> Slider (Sync)
            if self.analysis_vm.use_ref:
                # 0.5 -> 500
                slider_val = int(val_float * 1000)
                self.slider_thresh.setRange(0, 1000)
                self.slider_thresh.setValue(slider_val)
            else:
                # 4000 -> 4000
                slider_val = int(val_float)
                if slider_val > self.slider_thresh.maximum():
                    self.slider_thresh.setMaximum(slider_val + 100)
                self.slider_thresh.setValue(slider_val)

            txt = self.txt_mask_band.text().strip()
            rules = txt if (txt and txt.lower() != "mean") else None
            self.analysis_vm.set_mask_rules(rules)
            
            # Exclude Bands
            ex_str = self.txt_exclude.text()
            self.analysis_vm.set_exclude_bands(ex_str)
            
            # Chain
            chain = []
            for i in range(self.list_prep.count()):
                item = self.list_prep.item(i)
                if item.checkState() == Qt.Checked:
                    key = item.data(Qt.UserRole)
                    params = item.data(Qt.UserRole + 1)
                    if params is None: params = {}
                    chain.append({"name": key, "params": params})
            
            self.analysis_vm.set_preprocessing_chain(chain)
            
            # Trigger Viz (Manual update requires calling viz, since signal is suppressed)
            self.update_viz()
            
            # Update detail windows
            self.img_windows = [w for w in self.img_windows if w.isVisible()]
            for w in self.img_windows:
                w.update_view()
                
        except Exception as e:
            print(f"Param Error: {e}")
        finally:
            self.updating_from_ui = False

    def update_viz(self, item=None):
        if self.list_viz is None or self.figure is None:
            return

        checked_items = []
        try:
            for i in range(self.list_viz.count()):
                item = self.list_viz.item(i)
                if item.checkState() == Qt.Checked:
                    checked_items.append(item.data(Qt.UserRole))
        except RuntimeError: return

        if not checked_items:
            # Safety check for figure
            if self.figure:
                self.figure.clear()
                self.figure.text(0.5, 0.5, "Check files to see Spectrum", ha='center')
                if self.canvas: self.canvas.draw()
            return
            
        # ... remainder of update_viz Logic ... 
        # (Since I'm replacing a chunk, I must include the rest or ensure EndLine covers it appropriately)
        # Wait, update_viz is long. I shouldn't replace the whole thing if I don't need to.
        # I'll just replace the start.
        
        progress = QProgressDialog("Processing Spectra...", "Cancel", 0, len(checked_items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            self.figure.clear()
            gs = gridspec.GridSpec(1, 1)
            ax = self.figure.add_subplot(gs[0,0])
            
            self.current_waves = None
            
            global_max_val = 100
            
            for i, path in enumerate(checked_items):
                progress.setValue(i)
                QApplication.processEvents()
                if progress.wasCanceled(): break
                
                if path in self.main_vm.data_cache:
                    cube, _ = self.main_vm.data_cache[path]
                    mx = np.nanmax(cube)
                    if mx > global_max_val: global_max_val = mx
                
                spec, waves = self.analysis_vm.get_processed_spectrum(path)
                if spec is not None:
                    x_axis = waves
                    if x_axis is None or len(x_axis) != len(spec):
                        x_axis = np.arange(len(spec))
                        
                    self.current_waves = x_axis
                    ax.plot(x_axis, spec, label=os.path.basename(path), alpha=0.7)
            
            if not self.analysis_vm.use_ref and not self.analysis_vm.mask_rules:
                if self.slider_thresh:
                    current_max_slider = self.slider_thresh.maximum()
                    if global_max_val > current_max_slider:
                        self.slider_thresh.setMaximum(int(global_max_val))
            
            progress.setValue(len(checked_items))
            ax.legend()
            ax.grid(True)
            ax.format_coord = self.format_coord
            self.canvas.draw()
            
        finally:
            progress.close()

    def reload_data_cache(self):
        # Clear VM Cache
        self.main_vm.data_cache.clear()
        
        # Clear Windows
        self.img_windows = [w for w in self.img_windows if w.isVisible()]
        for w in self.img_windows:
            w.load_data() # Re-load
            w.update_view()
            
        # Refresh current viz
        self.update_viz()
        QMessageBox.information(self, "Reload", "Data Cache Cleared and Views Refreshed.")

    def on_scroll(self, event):
        ax = event.inaxes
        if not ax: return
        
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

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.canvas.draw_idle()

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
            
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            
            self.canvas.draw_idle()

    def format_coord(self, x, y):
        if hasattr(self, 'current_waves') and self.current_waves is not None:
             try:
                idx = (np.abs(self.current_waves - x)).argmin()
                return f"Band #{idx+1} ({x:.1f}nm), Int: {y:.2f}"
             except: pass
        return f"x={x:.1f}, y={y:.2f}"

    def open_image_window(self, item):
        path = item.data(Qt.UserRole)
        # Check if already open
        for w in self.img_windows:
            if w.isVisible() and w.path == path:
                 w.raise_()
                 return
                 
        win = ImageViewer(path, self.analysis_vm)
        win.show()
        self.img_windows.append(win)

    def close_all_windows(self):
        for w in self.img_windows: w.close()
