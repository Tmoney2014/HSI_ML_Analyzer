# viewmodels/ — MVVM State Layer

**Generated:** 2026-04-03 <!-- AI가 수정함: 날짜 갱신 -->

## OVERVIEW
Three ViewModels form a dependency chain. Instantiated once in `MainWindow.__init__` and injected into Views via constructor. Never instantiate VMs in service or model code.

## DEPENDENCY CHAIN

```
MainViewModel                        # Root — no dependencies
  └── AnalysisViewModel(main_vm)     # Depends on MainVM
        └── TrainingViewModel(main_vm, analysis_vm)  # Depends on both
```

## FILES

### `main_vm.py` — MainViewModel
Root state container. All other VMs depend on it.

| Property/Method | Type | Role |
|----------------|------|------|
| `file_groups` | `dict[str, list[str]]` | Group name → file paths |
| `group_colors` | `dict[str, str]` | Group name → hex color |
| `processing_mode` | `str` | `"Raw"` / `"Reflectance"` / `"Absorbance"` |
| `white_ref`, `dark_ref` | `str` | Paths to reference files |
| `cache_white`, `cache_dark` | `ndarray\|None` | Lazily loaded mean spectra |
| `data_cache` | `SmartCache` | L1 cache: path → `(cube, waves)` |
| `ensure_refs_loaded()` | method | Lazy-loads ref cubes into `cache_white/dark` |
| `reset_session()` | method | Full state reset for New Project |
| `set_processing_mode(mode)` | method | Emits `mode_changed` signal |
| `add/remove/rename_group()` | methods | Emit `files_changed` |
| `move_file_to_group()` | method | Moves file between groups, emits `files_changed` |

**Signals:** `files_changed`, `refs_changed`, `mode_changed(str)`, `save_requested`

### `SmartCache` (in main_vm.py)
Thread-safe LRU OrderedDict with memory guard.
- Config: `max_items` (default 20), `min_memory_gb` (default 1.0) from `settings.json`
- Evicts oldest entries when count limit or RAM threshold hit
- Uses `psutil` for memory checks (optional import — degrades gracefully)

### `analysis_vm.py` — AnalysisViewModel
Owns the preprocessing pipeline state and 3-level visualization cache.

| Property/Method | Type | Role |
|----------------|------|------|
| `threshold` | `float` | Background mask threshold |
| `mask_rules` | `str\|None` | Complex band rule string |
| `exclude_bands_str` | `str` | User input: `"1-5, 92"` |
| `prep_chain` | `list[dict]` | Active (enabled) preprocessing steps |
| `_full_state` | `list[dict]` | Full chain including disabled steps |
| `set_full_state(state)` | method | Set chain from UI (preserves disabled) |
| `get_processed_spectrum(path)` | method | Returns `(spectrum, waves)` using 3-level cache |
| `parse_exclude_bands()` | method | Parses exclude string → 0-based indices |

**Signals:** `visualization_updated(object, object)`, `error_occurred(str)`, `model_updated`, `params_changed`, `base_data_invalidated`

**Cache invalidation rules:**
- `threshold` / `mask_rules` change → clears `_cached_masked_mean` + emits `base_data_invalidated`
- `mode_changed` (from MainVM) → clears `_cached_ref_cube` + `_cached_masked_mean`
- File change → clears all 3 cache levels

### `training_vm.py` — TrainingViewModel
Orchestrates async training and optimization workers.

| Property/Method | Type | Role |
|----------------|------|------|
| `model_type` | `str` | `"Linear SVM"` / `"LDA"` / `"PLS-DA"` |
| `val_ratio` | `float` | Train/validation split |
| `n_features` | `int` | Target band count for SPA |
| `band_selection_method` | `str` | `'spa'` (default) / `'full'` — band selection strategy <!-- AI가 수정함: 누락된 속성 추가 --> |
| `best_n_features` | `int` | Optimization 완료 후 설정되는 최적 밴드 수; `'full'` 모드에서는 spinner 갱신 생략 <!-- AI가 수정함: 누락된 속성 추가 --> |
| `output_folder`, `model_name` | `str` | Path components; `full_output_path` property combines them |
| `excluded_files` | `set[str]` | Files excluded from training labels |
| `cached_base_data` | `dict[str, tuple]` | L3 cache: path → `(X_base, y)` |
| `run_training(...)` | method | Spawns `TrainingWorker` on `QThread` |
| `run_optimization(...)` | method | Spawns `OptimizationWorker` on `QThread` |
| `stop_training()` | method | Calls `worker.stop()` on active worker(s) |
| `get_config()` / `set_config()` | methods | Serialize/deserialize for project save/load |

**Signals:** `log_message(str)`, `progress_update(int)`, `training_finished(bool)`, `config_changed`

**Cache invalidation:** `base_data_invalidated` (from AnalysisVM), `refs_changed`, `files_changed`, `mode_changed` all → clear `cached_base_data`

## CONVENTIONS

- **Constructor injection only:** VMs never self-wire; `MainWindow.__init__` does all wiring
- **Signal debouncing:** `params_changed` triggers auto-save via `QTimer.singleShot` in MainWindow
- **`base_data_invalidated` vs `params_changed`:** Only threshold/mask changes invalidate L3 cache; Gap/prep chain changes do NOT (only invalidate L2 visualization)
- **Thread safety:** Never access `cached_base_data` from worker threads directly — workers emit `base_data_ready` signal, VM receives it on main thread
- **`band_selection_method == 'full'`:** `best_n_features` 속성이 설정되어도 `TabTraining.on_finished()`에서 `spin_bands` spinner 갱신을 건너뜀 — Full Band 모드에서는 밴드 수가 무의미하므로 <!-- AI가 수정함: Full Band guard 동작 명시 -->

## ANTI-PATTERNS

- Do not instantiate `AnalysisViewModel` or `TrainingViewModel` without their required VM dependencies
- Do not directly clear `cached_base_data` from View code — only TrainingVM manages this
- Do not connect to `params_changed` for base-data-heavy operations — use `base_data_invalidated`
- Do not access `main_vm.data_cache` from background threads without going through worker pattern
