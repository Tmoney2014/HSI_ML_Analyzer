# services/ — Business Logic Layer

**Generated:** 2026-04-03 <!-- AI가 수정함: 날짜 갱신 -->

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

Supported chain step names: `"SG"`, `"SimpleDeriv"`, `"SNV"`, `"L2"`, `"MinSub"`, `"MinMax"`.

### `band_selection_service.py`
```python
select_best_bands(
    data_cube: ndarray,   # (H,W,B) or (N,B)
    n_bands: int = 5,
    method: str = 'spa',  # 'spa' | 'full' | 'anova_f' | 'spa_lda_fast' | 'spa_lda_greedy' | 'lda_coef' <!-- AI가 수정함: supervised 방법 4개 추가 -->
    exclude_indices: list = None,
    keep_order: bool = False,
    labels: ndarray = None  # AI가 수정함: supervised 방법용 클래스 레이블 (anova_f/spa_lda_fast/spa_lda_greedy/lda_coef 필수)
) -> (selected_indices: list[int], importance: ndarray, mean_spectrum: ndarray)
```
Band selection service supporting both unsupervised (SPA, Full Band) and supervised methods (ANOVA-F, SPA-LDA Fast, SPA-LDA Greedy, LDA-coef). All methods subsample to `cfg_get('spa','max_samples',10000)`. Returns 0-based indices. Callers may pass `X.reshape(-1, 1, B)` — function handles both 2D and 3D input. <!-- AI가 수정함: supervised 방법 추가 반영 -->

**Supervised methods require `labels` parameter** — raises `ValueError` if `labels=None`.
- `anova_f`: sklearn `f_classif` F-통계량 기반 ranking. NaN/Inf → 0 처리. 클래스 수 < 2 시 `ValueError`.
- `spa_lda_fast`: SPA로 `max(n_bands*3, n_bands+2)` 후보 추출 → LDA coef ranking. 빠름.
- `spa_lda_greedy`: Greedy cross-validation (cv=min(3,min_class_count)). 느림 — UserWarning 발생. 최적화 루프 비권장.
- `lda_coef`: LDA(`solver='lsqr', shrinkage='auto'`) fit → `coef_` 절댓값 합 ranking. HSI처럼 n_samples < n_bands 환경에서 필수 solver 설정.

`importance` return value: SPA는 직교 기여도 노름, supervised 방법은 각 method의 score 배열. Used for `band_importance.png` visualization. <!-- AI가 수정함: supervised importance 의미 추가 -->

### `learning_service.py` — `LearningService` <!-- AI가 수정함: 지원 모델 목록 갱신 -->
```python
train_model(X, y, model_type="Linear SVM", test_ratio=0.2, log_callback=None)
    → (model, metrics: dict)
    # model_type: "Linear SVM" | "LDA" | "PLS-DA" | "Ridge Classifier" | "Logistic Regression"  # AI가 수정함
    # metrics keys: TrainAccuracy, TestAccuracy (0-100), F1Score, TotalSamples, TestSplit
```

**`export_model` key behaviors:**
- Computes `RequiredRawBands` by expanding `SelectedBands` with gap-adjacent bands needed at runtime (derivative gap/order + SG window radius)
- Sorts/deduplicates `SelectedBands` before export and reorders weight columns to match the exported feature order  <!-- AI가 수정함: band-order contract 보강 -->
- PLS-DA: `_train_pls` monkey-patches `model.export_coef_` and `model.export_intercept_` to normalize shapes before export — this is load-bearing, do not remove
- Preserves model-specific `Weights` / `Bias` shapes; downstream runtime must interpret them using `OriginalType` + `IsMultiClass` rather than assuming one universal tensor layout  <!-- AI가 수정함: model-aware shape contract 보강 -->
- Emits `PrepChainOrder` as the authoritative preprocessing replay order for runtime parity; documented fallback order is only for legacy/no-field cases  <!-- AI가 수정함: preprocessing order contract 보강 -->
- Exports `EstimatedFPS` as diagnostic metadata only; it must not be treated as a runtime-correctness input  <!-- AI가 수정함: metadata 성격 명시 -->
- All band indices in model.json are **0-based**

### `optimization_service.py` — `OptimizationService(QObject)` <!-- AI가 수정함: 3D 탐색 공간 확장 반영 -->
**Signals:** `log_message(str)`

```python
run_optimization(initial_params: dict, trial_callback: callable, band_methods=None) → (best_params, history)
lookahead_hill_climbing(start_val, step, lookahead, max_val, evaluator, ...) → (best_val, best_acc, best_params)
```
3D 전체 탐색: `band_method × n_bands × gap`. `band_methods=None` 이면 `initial_params['band_selection_method']` 단일값으로 폴백. `_TARGET_KEYS = ["SimpleDeriv"]` — prep_chain에 해당 step이 없으면 `gap_list = [0]` 로 단축. `trial_callback(params) → float score` 는 `OptimizationWorker`가 제공. Full Band 모드(`method='full'`)에서는 로그에 `Bands=Full Band` 출력. dedup key는 `(band_method, n_features, gap)` 3-tuple. <!-- AI가 수정함: 3D 루프 + gap 조건부 + dedup 3-tuple 명시 -->

### `training_worker.py` — `TrainingWorker(QObject)`
**Signals:** `progress_update(int)`, `log_message(str)`, `training_finished(bool)`, `base_data_ready(str, tuple)`

Full async training pipeline:
1. Validate ≥2 classes
2. Load files via `ProcessingService.get_base_data()` (cache-aware) — emits `base_data_ready(path, (X_base, None))` per file
3. Apply prep chain: `ProcessingService.apply_preprocessing_chain(X_base, prep_chain)`
4. Band selection: `select_best_bands(X.reshape(-1,1,B), n_features)`
5. Train: `LearningService.train_model(X_sub, y, model_type, test_ratio)`
6. Export: `LearningService.export_model(...)` → model.json

### `optimization_worker.py` — `OptimizationWorker(QObject)` <!-- AI가 수정함: band_methods 다중 파라미터 반영 -->
**Signals:** `progress_update(int)`, `log_message(str)`, `optimization_finished(bool)`, `data_ready(object, object)`, `base_data_ready(object, object)`

`__init__(band_methods=None)` — `self.band_methods: list` (None이면 `initial_params`에서 단일값 추출). `self.band_selection_method` 속성 제거됨. `_total_trials = len(band_methods) × len(band_range) × len(gap_range)`. `trial_callback`에서 `band_method = params.get('band_selection_method', 'spa')` 추출 → `_evaluate_cached_data(band_method=band_method)` 전달. `_evaluate_cached_data(prep_chain, n_features, band_method=None)` 시그니처. Preloads all base data once (`_prepare_base_data`), then passes `trial_callback` to `OptimizationService.run_optimization()`. `_prepare_base_data` uses `ProcessingService.get_base_data()` (mask + ref only, no prep chain). <!-- AI가 수정함: band_methods 다중 지원, _total_trials 3D 계산, _evaluate_cached_data 시그니처 -->

### `experiment_runner.py` — `ExperimentRunner` (순수 Python 클래스, QObject 아님) <!-- AI가 수정함: 4D 탐색 공간 추가 -->

```python
run_grid(
    band_methods: list[str],
    n_bands_list: list[int],
    model_types: list[str],
    gap_range: tuple[int,int] = (1, 40),  # SimpleDeriv 있을 때만 탐색
    stop_flag: threading.Event = None,
) → list[dict]
```

**4D 전체 탐색:** `band_method × n_bands × gap × model_type`. SimpleDeriv 감지: prep_chain에 `_TARGET_KEYS` step이 있으면 `gap_range` 전체 탐색, 없으면 `gap_list = [0]`. `_prep_cache` 딕셔너리로 gap별 전처리 결과 캐시.

**15컬럼 CSV:** `_CSV_FIELDNAMES` — `band_method, n_bands, gap(index 3), model_type, train_acc, test_acc, f1, ...`

**PNG prefix:** `g{gap_val}_` (gap 포함 파일명 구분).

**4레벨 stop 전파:** `stop_flag` 체크가 method → n_bands → gap → model_type 4중 루프 전체에서 동작.

**`_best_per_bm_mt(results, metric="test_acc") → (best_results, best_config_map)`** (static):
- `(band_method, model_type)` 조합별 metric 최고 trial 1개 선택, 에러 trial 자동 제외.

**`_write_paper_summary()`:** `_best_results` 기반 heatmap 3개 + "Best Configuration per (Band Method, Model)" 테이블 포함. <!-- AI가 수정함: 4D 시그니처, gap 조건부, 15컬럼, _best_per_bm_mt, Best Config 테이블 명시 -->

**⚠️ CRITICAL:** `ExperimentRunner`는 순수 Python 클래스. `QObject` 상속 금지. UI/VM import 금지.

### `experiment_worker.py` — `ExperimentWorker(QObject)` <!-- AI가 수정함: n_bands_list/gap_range 파라미터 반영 -->
**Signals:** `progress_update(int)`, `log_message(str)`, `training_finished(bool)`, `base_data_ready(object, object)`

`params` dict에서 `n_bands_list` (없으면 `n_bands` 단일값 폴백) + `gap_range=(1,1)` 읽음. `_total_trials_est = len(band_methods) × len(n_bands_list) × gap_count × len(model_types)`. `[Trial]` 로그 prefix 감지 → `progress_update.emit`. 중복 CM(confusion matrix) 블록 없음 — `ExperimentRunner.run_grid`가 per-trial CM 저장. <!-- AI가 수정함: n_bands_list backward compat, gap_range, 진행률 emit 방식 명시 -->

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
