# views/dialogs/ — Modal Dialogs

**Generated:** 2026-04-02

## OVERVIEW
Two standalone dialogs. Neither holds a ViewModel. Pure UI input collectors.

## FILES

| File | Class | Base | VM | Role |
|------|-------|------|----|------|
| `preprocessing_dialog.py` | `PreprocessingSettingsDialog` | `QDialog` | none | Collects params for one preprocessing step |
| `project_welcome_dialog.py` | `ProjectWelcomeDialog` | `QDialog` | none | Startup New / Open choice |

---

## PreprocessingSettingsDialog

Opened by `TabAnalysis.open_prep_settings()` on prep-list item double-click.

### Constructor
```python
PreprocessingSettingsDialog(step_key: str, current_params: dict, parent=None)
```
Renders different widgets per `step_key`:
- `"SG"` → `window_size` (int), `poly_order` (int), `deriv` (int 0–2)
- `"SimpleDeriv"` → `gap` (int), `order` (int), `apply_ratio` (bool), `ndi_threshold` (float)
- `"3PointDepth"` → `gap` (int)

### Usage pattern
```python
dlg = PreprocessingSettingsDialog(step_key, current_params, parent=self)
if dlg.exec_() == QDialog.Accepted:
    new_params = dlg.get_params()   # dict with updated values
```

### Signals
- `btns.accepted → self.accept()`
- `btns.rejected → self.reject()`

---

## ProjectWelcomeDialog

Modal startup dialog; shown by `MainWindow` via `QTimer.singleShot` on launch.

### Usage
```python
dlg = ProjectWelcomeDialog(parent=self)
dlg.exec_()
choice = dlg.result   # 'new' | 'open' | None (dismissed)
```

### Signals
- `btn_new.clicked → self.on_new()` → sets `self.result = 'new'`, calls `accept()`
- `btn_open.clicked → self.on_open()` → sets `self.result = 'open'`, calls `accept()`

---

## CONVENTIONS
- Always use `exec_()` (modal); never `show()` (modeless) for these dialogs
- Read result attributes / call `get_params()` only after `exec_()` returns
- Do NOT add business logic — dialogs return raw param dicts / choices only
