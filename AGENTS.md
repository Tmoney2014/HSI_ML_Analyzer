# HSI ML Analyzer - Knowledge Base

**Generated:** 2026-03-10
**Commit:** 98a35c4
**Branch:** main
**Project:** HyperSort - 고속 HSI 플라스틱 선별 시스템

## OVERVIEW

Python/PyQt5 ML application for offline training of HSI (Hyperspectral Imaging) plastic sorting models. Outputs `model.json` for C# WPF runtime (700 FPS target). Part 1 of a two-part system; Part 2 is a separate C# WPF runtime repo.

## STRUCTURE

```
HSI_ML_Analyzer/
├── docs/                    # Design specs (data pipeline, runtime, cache hierarchy)
├── Python_Analysis/         # Main application (MVVM, PyQt5)
│   ├── main.py              # Entry point — sets sys.path, launches MainWindow
│   ├── config.py            # ConfigManager singleton; public API: get(), set_value(), save()
│   ├── settings.json        # Runtime tunables (cache, training, SPA, optimization)
│   ├── default_project.json # Template for new project state
│   ├── Gemini.md            # Detailed AI context (Korean) — read for deep system design
│   ├── models/              # Pure math functions (C# parity required)
│   ├── services/            # Business logic + QThread workers
│   ├── viewmodels/          # MVVM state (MainVM → AnalysisVM → TrainingVM dependency chain)
│   └── views/               # PyQt5 widgets, tabs, dialogs, components
└── requirements.txt
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Run app | `Python_Analysis/main.py` | PyQt5 entry point |
| Config read/write | `Python_Analysis/config.py` | `get('section.key', default=X)` |
| Data loading | `services/data_loader.py` | ENVI .hdr/.raw → numpy (H,W,B) |
| Masking | `models/processing.py::create_background_mask` | Supports complex string rules with `&`, `\|` |
| Preprocessing functions | `models/processing.py` | Pure functions — C# runtime parity required |
| Preprocessing pipeline | `services/processing_service.py` | Orchestrates mode conversion + prep chain |
| Band selection (SPA) | `services/band_selection_service.py` | Reduces 154 → N orthogonal bands |
| ML training + export | `services/learning_service.py` | LDA/SVM/PLS-DA; `export_model()` → model.json |
| Auto-ML | `services/optimization_service.py` | Optimizes Gap, NDI threshold, band count |
| Async training | `services/training_worker.py` | QThread wrapper; emits `base_data_ready` for caching |
| Async optimization | `services/optimization_worker.py` | QThread wrapper for Auto-ML |
| Global file/ref state | `viewmodels/main_vm.py::MainViewModel` | SmartCache LRU, group management, ref loading |
| Analysis state | `viewmodels/analysis_vm.py::AnalysisViewModel` | Prep chain, masking params, 3-level cache |
| Training state | `viewmodels/training_vm.py::TrainingViewModel` | Worker orchestration, base_data_cache |
| Main window | `views/main_window.py` | 3-tab layout, project save/load, startup dialog |
| Data tab | `views/tabs/tab_data.py` | File group management, ref assignment |
| Analysis tab | `views/tabs/tab_analysis.py` | Spectrum visualization, preprocessing UI |
| Training tab | `views/tabs/tab_training.py` | Training config, results, optimization trigger |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `MainViewModel` | class | `viewmodels/main_vm.py` | Root VM; holds file_groups, SmartCache, refs |
| `AnalysisViewModel` | class | `viewmodels/analysis_vm.py` | Analysis state; 3-level spectrum cache |
| `TrainingViewModel` | class | `viewmodels/training_vm.py` | Training orchestration; cached_base_data dict |
| `SmartCache` | class | `viewmodels/main_vm.py` | Thread-safe LRU+memory-limit OrderedDict |
| `ProcessingService` | class | `services/processing_service.py` | Central preprocessing pipeline (single source) |
| `LearningService` | class | `services/learning_service.py` | ML training factory + model.json export |
| `OptimizationService` | class | `services/optimization_service.py` | Auto-ML: Gap/NDI/band optimization |
| `TrainingWorker` | class | `services/training_worker.py` | QThread for async training |
| `OptimizationWorker` | class | `services/optimization_worker.py` | QThread for async Auto-ML |
| `select_best_bands` | func | `services/band_selection_service.py` | SPA orthogonal band selection |
| `create_background_mask` | func | `models/processing.py` | Mask from threshold or string rules |
| `apply_simple_derivative` | func | `models/processing.py` | Gap-Diff (Log-Gap-Diff core); C# parity |
| `apply_rolling_3point_depth` | func | `models/processing.py` | Continuum Removal Lite; C# parity |
| `apply_absorbance` | func | `models/processing.py` | -log10(R) transform; C# parity |
| `ConfigManager` | class | `config.py` | Singleton; `get('section.key')` dotted access |

## DATA FLOW

```
ENVI file (.hdr/.raw)
  → data_loader.load_hsi_data() → (H,W,B) numpy cube
  → MainViewModel.data_cache [SmartCache L1]
  → create_background_mask()  (Raw DN values — always mask on Raw)
  → ProcessingService.convert_to_ref() [Raw → Ref → Abs]
  → ProcessingService.process_cube() [SG, Gap-Diff, 3PointDepth]
  → TrainingWorker: base_data_cache [L2 per-file]
  → select_best_bands() [SPA: N→5 bands]
  → LearningService.train_model() [LDA/SVM/PLS-DA]
  → LearningService.export_model() → model.json
```

## model.json STRUCTURE

```json
{
  "ModelType": "LinearModel",
  "OriginalType": "LinearDiscriminantAnalysis",
  "SelectedBands": [40, 42, 97, 99, 105],
  "RequiredRawBands": [35, 40, 42, 92, 97, 99, 100, 105],
  "Weights": [[...], [...]],
  "Bias": [...],
  "Preprocessing": { "ApplySG": false, "ApplyDeriv": true, "Gap": 5 },
  "Labels": { "0": "PP", "1": "PE" }
}
```

## CONVENTIONS

- **Architecture:** MVVM — Views receive VMs via constructor injection in `MainWindow.__init__`
- **Threading:** Any operation >100ms → `*_worker.py` (QThread). Never in VM or View directly.
- **Config access:** `from config import get; get('section', 'key', default)` or `get('section.key')`
- **Runtime parity:** `models/processing.py` functions must match C# runtime exactly — verify after any change
- **ProcessingService is single source:** Never duplicate preprocessing logic outside `processing_service.py`
- **Masking always on Raw:** `create_background_mask` uses Raw DN values, not converted data
- **Strict Mode:** Preprocessing functions raise `ValueError` on insufficient bands (not silent partial results)
- **`/// <ai>AI가 작성함</ai>`** docstring tag marks AI-written code blocks for review
- **`# AI가 수정함:`** inline comment marks AI-modified logic

## CACHE LAYERS

| Layer | Variable | Contents | Invalidated by |
|-------|----------|----------|----------------|
| L1 | `MainViewModel.data_cache` | Raw cubes (H,W,B) | File removal |
| L2 | `AnalysisViewModel._cached_masked_mean` | Masked+converted mean | threshold, mask_rules, mode change |
| L3 | `TrainingViewModel.cached_base_data` | Per-file (X_base, y) | refs_changed, files_changed, base_data_invalidated |

## ANTI-PATTERNS

- Do not import from `views` or `viewmodels` into `services` or `models`
- Do not run `LearningService`, `SPA`, or `ProcessingService.process_cube()` on the main thread
- Do not use global state for HSI cubes — pass via `DataRepository`/`data_cache`
- Do not hardcode paths — use `config.get()` or constructor arguments
- Do not call `apply_snv()` in production pipelines (requires full-frame stats, MROI incompatible)
- Do not commit `Python_Analysis/venv/` to version control
- Do not modify `models/processing.py` without verifying C# runtime parity

## COMMANDS

```bash
# Run application
cd Python_Analysis
python main.py

# Batch launcher
Python_Analysis/run_gui.bat

# Install dependencies
pip install -r requirements.txt
```

## NOTES

- `processing_mode`: `"Raw"` | `"Reflectance"` | `"Absorbance"` (set on `MainViewModel`)
- `Ignore` group is reserved — excluded from training labels automatically
- `RequiredRawBands` in model.json = SelectedBands + gap-adjacent bands needed for Gap-Diff at C# runtime
- `docs/` contains Korean-language specs for data pipeline, C# inference runtime, and cache hierarchy
