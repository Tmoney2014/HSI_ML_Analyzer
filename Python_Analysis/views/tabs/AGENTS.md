# Views / Tabs Layer

**Generated:** 2026-04-03 | **Commit:** be1e3ae <!-- AI가 수정함: 날짜·커밋 갱신 -->

## FILES

| File | Class(es) | LOC | VM(s) |
|------|-----------|-----|-------|
| `tab_data.py` | `ClassListWidget`, `TabData` | 398 | `MainViewModel` | <!-- AI가 수정함: LOC 418→398 (실측값) -->
| `tab_analysis.py` | `TabAnalysis` | 610 | `MainViewModel` + `AnalysisViewModel` | <!-- AI가 수정함: LOC 652→610 (실측값) -->
| `tab_training.py` | `TabTraining` | 675 | `TrainingViewModel` | <!-- AI가 수정함: LOC 316→675 (4D 탐색 UI 확장) -->

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

## tab_analysis.py — Central analysis tab (610 lines) <!-- AI가 수정함: LOC 652→610 -->

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

## tab_training.py <!-- AI가 수정함: 4D 탐색 UI 확장 반영 -->

### Key state
```python
self.vm: TrainingViewModel
self.tree_files: QTreeWidget    # groups as top-level, files as children with checkboxes
self.log_text: QTextEdit        # green-on-black console style
# Optimize 섹션
self.lbl_spa_greedy_warning: QLabel        # SPA-LDA Greedy 느림 경고
# Experiment 섹션
self.list_experiment_n_bands: QListWidget  # 체크박스 목록 (5,10,15,20,25,30,35,40)
self.spin_experiment_gap_min: QSpinBox     # gap 탐색 최솟값
self.spin_experiment_gap_max: QSpinBox     # gap 탐색 최댓값
self.lbl_experiment_spa_greedy_warning: QLabel  # Experiment SPA-LDA Greedy 경고
self.lbl_experiment_gap_disabled: QLabel   # SimpleDeriv 없을 때 gap 비활성 안내
```

### init_from_vm_state() — VM→UI sync (called by MainWindow on load/new)
Blocks all widget signals during sync to prevent `_on_ui_changed` feedback loop:
```python
self.blockSignals(True)
# set txt_folder, txt_name, txt_desc, combo_model, spin_ratio, spin_bands from vm props
# _restore_checked_n_bands(vm.experiment_n_bands_list)
# spin_experiment_gap_min/max from vm.experiment_gap_min/max
self.blockSignals(False)
self.refresh_file_tree()
```

### _on_ui_changed() — UI→VM sync
Connected to: `txt_name.textChanged`, `txt_desc.textChanged`, `combo_model.currentIndexChanged`, `spin_ratio.valueChanged`, `spin_bands.valueChanged`.
Writes directly to `vm.output_folder/model_name/model_desc/model_type/val_ratio/n_features` → emits `vm.config_changed` → triggers debounced auto-save in `MainWindow`.

### _on_band_method_changed() <!-- AI가 수정함: 신규 핸들러 -->
`combo_band_method.currentIndexChanged` 연결. `spa_lda_greedy` 선택 시 `lbl_spa_greedy_warning` 표시.

### _update_gap_widgets_enabled() <!-- AI가 수정함: 신규 핼퍼 -->
현재 prep_chain(`analysis_vm.prep_chain`)에 `SimpleDeriv`가 있는지 감지.
- 있으면: `spin_experiment_gap_min/max` 활성, `lbl_experiment_gap_disabled` 숨김
- 없으면: spinbox 비활성(value=0), `lbl_experiment_gap_disabled` 표시

### _update_experiment_summary() <!-- AI가 수정함: 신규 핼퍼 -->
4D trial 수 공식: `len(checked_band_methods) × len(checked_n_bands) × gap_count × len(checked_model_types)` 을 요약 레이블에 표시.

### _get_checked_n_bands() / _restore_checked_n_bands(n_bands_list) <!-- AI가 수정함: 신규 핼퍼 -->
`list_experiment_n_bands`의 체크된 항목을 `list[int]`로 반환 / 복원.

### refresh_file_tree()
Reads `vm.main_vm.file_groups` → builds `QTreeWidget` with tristate group nodes.
Files in `vm.excluded_files` → unchecked; others → checked.

### on_tree_item_changed(item, column)
File checkboxes only (items with `Qt.UserRole` path set).
`unchecked` → `vm.set_file_excluded(path, True)`; `checked` → `vm.set_file_excluded(path, False)`.

### Training / Optimization / Experiment flow <!-- AI가 수정함: Experiment 항목 추가 -->
- `on_start_click()` → clears log → `vm.run_training()` (no args — VM is source of truth)
- `on_optimize_click()` → clears log → `vm.run_optimization()`
- `on_export_click()` → `_get_checked_n_bands()` + `spin_experiment_gap_min/max` 수집 → `vm.run_experiment_grid(n_bands_list=..., gap_min=..., gap_max=...)` <!-- AI가 수정함: Experiment 실행 흐름 추가 -->
- All three disable buttons via `set_buttons_enabled(False)` during run
- `on_finished(success)`: re-enables buttons; if `vm.best_n_features` is set **and** `vm.band_selection_method != 'full'`, syncs `spin_bands` and clears it (one-shot); Full Band 모드에서는 spinner 갱신 건너뜀; `vm.best_band_method` 가 있으면 `combo_band_method` 자동 업데이트 후 클리어 (one-shot) <!-- AI가 수정함: best_band_method 소비 로직 추가 -->

### Signals consumed
- `vm.log_message` → `append_log(msg)` (green console)
- `vm.training_finished(bool)` → `on_finished(success)`
