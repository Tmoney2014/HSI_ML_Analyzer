# Views / Tabs Layer

**Generated:** 2026-03-10 | **Commit:** 98a35c4

## FILES

| File | Class(es) | LOC | VM(s) |
|------|-----------|-----|-------|
| `tab_data.py` | `ClassListWidget`, `TabData` | 418 | `MainViewModel` |
| `tab_analysis.py` | `TabAnalysis` | 652 | `MainViewModel` + `AnalysisViewModel` |
| `tab_training.py` | `TabTraining` | 305 | `TrainingViewModel` |

---

## tab_data.py

### ClassListWidget
- Subclasses `QListWidget`; accepts drops from other `QListWidget` instances
- `dropEvent`: extracts file paths from dragged items → calls `vm.move_file_to_group(path, target_group)`
- Target group extracted from item text via `item.text().split(" (")[0]`

### TabData
- Signals consumed: `vm.files_changed` → `_refresh_lists()`, `vm.refs_changed` → `_refresh_refs()`
- `_refresh_lists()`: clears + rebuilds `list_groups`; stores raw group name in `item.data(Qt.UserRole)` (display text includes count); preserves current selection
- Groups named `"-"`, `"unassigned"`, `"trash"`, `"ignore"` (case-insensitive) → shown with `🚫` prefix, gray text, excluded from training
- `on_group_selected()`: populates `list_files` from `vm.file_groups[name]`; enables add/remove buttons
- File drag: `list_files` is `DragOnly`; groups list (`ClassListWidget`) is `DropOnly` — drag files → drop on group to move
- `restore_ui()`: calls `_refresh_lists()` + `_refresh_refs()` + syncs mode radio buttons from `vm.processing_mode`
- Mode guard: switching to Reflectance/Absorbance requires `vm.white_ref` to exist; resets to Raw with warning if missing

---

## tab_analysis.py — Central analysis tab (652 lines)

### Key state
```python
self.img_windows: list[ImageViewer]  # open detail windows; pruned on each update
self.updating_from_ui: bool          # guard against signal loops
self.list_prep: QListWidget          # drag-reorderable preprocessing chain
self.slider_thresh / txt_thresh      # background cutoff; synced bidirectionally
```

### update_params() — central UI→VM sync
Called by: threshold text `returnPressed`, mask rules `Apply` button, slider `sliderReleased`, prep list `itemChanged` / `rowsMoved`, "Update All Graphs" button.

Flow:
1. Read `txt_thresh` → `analysis_vm.set_threshold()`
2. Sync slider position
3. Read `txt_mask_band` → `analysis_vm.set_mask_rules()`
4. Read `txt_exclude` → `analysis_vm.set_exclude_bands()`
5. Build `full_state` list from all `list_prep` items (order + enabled + params) → `analysis_vm.set_full_state()`
6. Call `update_viz()` + refresh all open `ImageViewer` windows

### restore_ui() — VM→UI sync (called by MainWindow on load/new)
- Restores threshold slider + text from `analysis_vm.threshold`
- Restores mask rules, exclude bands
- Rebuilds `list_prep` from `analysis_vm.get_full_state()` (preferred) or `prep_chain` (legacy fallback)
- Appends any default steps missing from saved state (handles app version upgrades)
- Calls `update_viz()` at end

### on_model_updated() — external VM change → UI sync
Connected to `analysis_vm.model_updated` signal (emitted by Auto-ML optimization).
Blocks signals, updates threshold + prep list checked/param state, calls `update_viz()`.

### Prep list item data slots
- `Qt.UserRole` → step key string (e.g. `"SimpleDeriv"`, `"SG"`, `"3PointDepth"`)
- `Qt.UserRole + 1` → params dict (e.g. `{"gap": 5, "order": 1, "ratio": False, "ndi_threshold": 1000.0}`)
- `_update_item_label(item)` — rebuilds display text with param summary inline

### Slider behavior
- `sliderValueChanged` → text update only (visual feedback during drag)
- `sliderReleased` → calls `update_params()` (heavy update)
- Range: Raw mode `0–65535`; Ref/Abs mode `0–1000` (maps to `0.0–1.0`)

### ImageViewer lifecycle
- Double-click on file list item → `open_image_window(item)` → creates `ImageViewer(path, analysis_vm)`
- Deduplication: skips if same path already visible
- `close_all_windows()` called by `MainWindow.closeEvent`

---

## tab_training.py

### Key state
```python
self.vm: TrainingViewModel
self.tree_files: QTreeWidget    # groups as top-level, files as children with checkboxes
self.log_text: QTextEdit        # green-on-black console style
```

### init_from_vm_state() — VM→UI sync (called by MainWindow on load/new)
Blocks all widget signals during sync to prevent `_on_ui_changed` feedback loop:
```python
self.blockSignals(True)
# set txt_folder, txt_name, txt_desc, combo_model, spin_ratio, spin_bands from vm props
self.blockSignals(False)
self.refresh_file_tree()
```

### _on_ui_changed() — UI→VM sync
Connected to: `txt_name.textChanged`, `txt_desc.textChanged`, `combo_model.currentIndexChanged`, `spin_ratio.valueChanged`, `spin_bands.valueChanged`.
Writes directly to `vm.output_folder/model_name/model_desc/model_type/val_ratio/n_features` → emits `vm.config_changed` → triggers debounced auto-save in `MainWindow`.

### refresh_file_tree()
Reads `vm.main_vm.file_groups` → builds `QTreeWidget` with tristate group nodes.
Files in `vm.excluded_files` → unchecked; others → checked.

### on_tree_item_changed(item, column)
File checkboxes only (items with `Qt.UserRole` path set).
`unchecked` → `vm.set_file_excluded(path, True)`; `checked` → `vm.set_file_excluded(path, False)`.

### Training / Optimization flow
- `on_start_click()` → clears log → `vm.run_training()` (no args — VM is source of truth)
- `on_optimize_click()` → clears log → `vm.run_optimization()`
- Both disable buttons via `set_buttons_enabled(False)` during run
- `on_finished(success)`: re-enables buttons; if `vm.best_n_features` is set, syncs `spin_bands` and clears it (one-shot)

### Signals consumed
- `vm.log_message` → `append_log(msg)` (green console)
- `vm.training_finished(bool)` → `on_finished(success)`
