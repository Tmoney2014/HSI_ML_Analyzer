# services/ — Business Logic Layer

**Generated:** 2026-03-10

## OVERVIEW
Orchestrates the full HSI training pipeline. Services are UI-agnostic; workers are QObject subclasses moved to QThread for >100ms operations.

## FILES

### `data_loader.py`
```python
load_hsi_data(header_path: str) -> (cube: ndarray(H,W,B), wavelengths: list[float])
```
ENVI .hdr/.raw reader via `spectral.io.envi`. Falls back to `range(B)` if wavelengths missing. Raises `FileNotFoundError` on missing .hdr.

### `processing_service.py` — `ProcessingService` (static methods)
Central pipeline hub. **Single source of truth for all preprocessing.** Never call `models/processing.py` functions directly from workers or VMs — go through here.

| Method | Signature | Purpose |
|--------|-----------|---------|
| `process_cube` | `(cube, mode, threshold, mask_rules, prep_chain, white_ref=None, dark_ref=None) → (ndarray(N,B'), mask(H,W))` | Full pipeline: mask → ref convert → prep chain |
| `get_base_data` | `(cube, mode, threshold, mask_rules, white_ref, dark_ref) → (ndarray(N,B), mask)` | Mask + ref only — NO prep chain (for L3 cache) |
| `convert_to_ref` | `(cube, white_ref, dark_ref) → ndarray(H,W,B)` | Raw→Reflectance on 3D cube |
| `convert_to_ref_flat` | `(flat, white_ref, dark_ref) → ndarray(N,B)` | Raw→Reflectance on flattened pixels |
| `apply_preprocessing_chain` | `(flat_data, prep_chain) → ndarray(N,B')` | Applies chain list; raises `ValueError` on missing required params |

Supported chain step names: `"SG"`, `"SimpleDeriv"`, `"3PointDepth"`, `"SNV"`, `"L2"`, `"MinSub"`, `"MinMax"`, `"Center"`.

### `band_selection_service.py`
```python
select_best_bands(
    data_cube: ndarray,   # (H,W,B) or (N,B)
    n_bands: int = 5,
    method: str = 'spa',  # 'spa' | 'variance'
    exclude_indices: list = None,
    keep_order: bool = False
) -> (selected_indices: list[int], importance: ndarray, mean_spectrum: ndarray)
```
SPA orthogonal projection band selection. Subsamples to `cfg_get('spa','max_samples',10000)`. Returns 0-based indices. Callers may pass `X.reshape(-1, 1, B)` — function handles both 2D and 3D input.

### `learning_service.py` — `LearningService`
```python
train_model(X, y, model_type="Linear SVM", test_ratio=0.2, log_callback=None)
    → (model, metrics: dict)
    # metrics keys: TrainAccuracy, TestAccuracy (0-100), F1Score, TotalSamples, TestSplit

export_model(model, selected_bands, output_path, preprocessing_config=None,
             processing_mode="Raw", ...) → None
    # Writes model.json + band_importance.png at output_path
```

**`export_model` key behaviors:**
- Computes `RequiredRawBands` by expanding `SelectedBands` with gap-adjacent bands needed at runtime (derivative gap/order + SG window radius)
- PLS-DA: `_train_pls` monkey-patches `model.export_coef_` and `model.export_intercept_` to normalize shapes before export — this is load-bearing, do not remove
- All band indices in model.json are **0-based**

### `optimization_service.py` — `OptimizationService(QObject)`
**Signals:** `log_message(str)`

```python
run_optimization(initial_params: dict, trial_callback: callable) → (best_params, history)
lookahead_hill_climbing(start_val, step, lookahead, max_val, evaluator, ...) → (best_val, best_acc, best_params)
```
Grid searches Band Count × Gap Size; fine-tunes NDI threshold via `lookahead_hill_climbing`. `trial_callback(params) → float score` is provided by `OptimizationWorker`.

### `training_worker.py` — `TrainingWorker(QObject)`
**Signals:** `progress_update(int)`, `log_message(str)`, `training_finished(bool)`, `base_data_ready(str, tuple)`

Full async training pipeline:
1. Validate ≥2 classes
2. Load files via `ProcessingService.get_base_data()` (cache-aware) — emits `base_data_ready(path, (X_base, None))` per file
3. Apply prep chain: `ProcessingService.apply_preprocessing_chain(X_base, prep_chain)`
4. Band selection: `select_best_bands(X.reshape(-1,1,B), n_features)`
5. Train: `LearningService.train_model(X_sub, y, model_type, test_ratio)`
6. Export: `LearningService.export_model(...)` → model.json

### `optimization_worker.py` — `OptimizationWorker(QObject)`
**Signals:** `log_message(str)`, `optimization_finished(bool)`, `base_data_ready(str, tuple)`

Preloads all base data once (`_prepare_base_data`), then passes `trial_callback` to `OptimizationService.run_optimization()`. `trial_callback` applies prep chain + SPA + trains per candidate parameter set.

## CACHE HANDSHAKE (CRITICAL)

Workers emit `base_data_ready(file_path, (X_base, None))` after loading each file. `TrainingViewModel.on_base_data_ready` receives this on the main thread and stores it in `cached_base_data`. On subsequent runs, workers receive `base_data_cache` dict in their constructor and skip re-loading files that are already cached.

## CONVENTIONS
- UI-agnostic: no `PyQt5.QtWidgets` or `viewmodels` imports in service/model code
- `ProcessingService` is the single entry point for all preprocessing — never call `models/processing.py` functions directly from workers
- `ValueError` from prep chain = strict mode violation (insufficient bands) — let it propagate
- Worker constructors receive VM state snapshots (not live references) to avoid race conditions

## ANTI-PATTERNS
- Running `LearningService.train_model` or `SPA` on the main thread
- Importing from `views` or `viewmodels`
- Hardcoding paths — use constructor arguments
- Passing live PyQt widget references into service methods
