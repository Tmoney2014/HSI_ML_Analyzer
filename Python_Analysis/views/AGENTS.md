# Views Layer

**Generated:** 2026-04-03 | **Commit:** be1e3ae <!-- AI가 수정함: 날짜·커밋 갱신, tab_analysis LOC 651→610 -->

## STRUCTURE

```
views/
├── main_window.py                  # QMainWindow; sole VM creation + injection point
├── tabs/
│   ├── tab_data.py                 # Group/file management, ref setup, mode toggle
│   ├── tab_analysis.py             # Spectrum viz, preprocessing config (610 lines) <!-- AI가 수정함: LOC 651→610 -->
│   └── tab_training.py             # Training config, log, file tree
├── components/
│   ├── image_viewer.py             # Single-file HSI image + pixel spectrum viewer
│   └── custom_toolbar.py           # Matplotlib toolbar subclass (16 lines)
└── dialogs/
    ├── preprocessing_dialog.py     # Configure SG/SimpleDeriv/3PointDepth params
    └── project_welcome_dialog.py   # Startup "New / Open" choice dialog
```

## FILE → CLASS MAP

| File | Class | VM(s) | Key Signals Consumed |
|------|-------|-------|----------------------|
| `main_window.py` | `MainWindow` | Creates all 3 VMs | `files_changed`, `refs_changed`, `params_changed`, `config_changed`, `save_requested` |
| `tabs/tab_data.py` | `ClassListWidget`, `TabData` | `MainViewModel` | `files_changed`, `refs_changed` |
| `tabs/tab_analysis.py` | `TabAnalysis` | `MainViewModel` + `AnalysisViewModel` | `files_changed`, `mode_changed`, `error_occurred`, `model_updated` |
| `tabs/tab_training.py` | `TabTraining` | `TrainingViewModel` | `log_message`, `training_finished` |
| `components/image_viewer.py` | `ImageViewer` | `AnalysisViewModel` (read-only) | — (pulls state via attribute access) |
| `dialogs/preprocessing_dialog.py` | `PreprocessingSettingsDialog` | none | — |
| `dialogs/project_welcome_dialog.py` | `ProjectWelcomeDialog` | none | — |

## KEY PATTERNS

### VM Creation (MainWindow only)
```python
# MainWindow.__init__ — the ONLY place VMs are instantiated
self.main_vm = MainViewModel()
self.analysis_vm = AnalysisViewModel(self.main_vm)
self.training_vm = TrainingViewModel(self.main_vm, self.analysis_vm)
# Then injected into views via constructor:
self.tab_data = TabData(self.main_vm)
self.tab_analysis = TabAnalysis(self.main_vm, self.analysis_vm)
self.tab_training = TabTraining(self.training_vm)
```

### VM→UI Sync after project load / new project
- `tab_analysis.restore_ui()` — reads full VM state → resets all controls; called by `MainWindow` after `load_project_file` or `new_project`
- `tab_training.init_from_vm_state()` — reads VM config properties → sets all spin/combo/text widgets with signals blocked
- `tab_data.restore_ui()` — refreshes group list + refs + mode radio; called on new project

### Auto-save (debounced)
- `main_vm.files_changed` / `refs_changed` / `analysis_vm.params_changed` → `MainWindow.auto_save_slot()` (immediate)
- `training_vm.config_changed` → `MainWindow._auto_save_debounced()` → `QTimer` 1 s debounce → `_on_save_timer()`
- Guard: `self.is_project_active = False` during init/load prevents spurious saves

### Project load guard
`is_project_active` flag on `MainWindow` is set `False` at start of `load_project_file`/`new_project`, set `True` only on success. All auto-save slots check this flag first.

## ImageViewer — MVVM Deviation

`ImageViewer` imports `ProcessingService`, `load_hsi_data`, and `models.processing` directly:
```python
from services.data_loader import load_hsi_data
from models import processing
from services.processing_service import ProcessingService
```
This is an **allowed exception** for single-pixel spectrum visualization. The class is a standalone popup window, not part of the main data flow. Heavy ops (training, optimization) still go through VM-managed workers.

## ANTI-PATTERNS

- Instantiating VMs anywhere except `MainWindow.__init__`
- Calling `restore_ui()` / `init_from_vm_state()` before `__init__` completes (guard: check widget existence first)
- Triggering `_on_ui_changed()` or `update_params()` while signals are not blocked during `init_from_vm_state()` / `restore_ui()` — causes `params_changed` → auto-save before project is active
- Running preprocessing or data loading in view event handlers — use VM methods or workers
