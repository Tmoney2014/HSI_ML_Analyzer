# Python_Analysis - Development Guide

**Generated:** 2026-03-10 | **Commit:** 98a35c4

## OVERVIEW
Core PyQt5 application for training hardware-aware HSI sorting models. MVVM architecture with strict layer separation.

## STRUCTURE
```
Python_Analysis/
├── main.py              # Entry point; sets sys.path, Fusion style, launches MainWindow
├── config.py            # ConfigManager singleton — dotted key access, read/write settings.json
├── settings.json        # Runtime tunables: cache, training, SPA, optimization defaults
├── default_project.json # Template loaded on "New Project"
├── Gemini.md            # Korean-language deep-dive context doc
├── models/              # Pure math functions — C# parity required, no side effects
├── services/            # Business logic + QThread workers — UI-agnostic
├── viewmodels/          # MVVM state (MainVM → AnalysisVM → TrainingVM dependency chain)
└── views/               # PyQt5 widgets, 3-tab layout, dialogs, components
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Config read | `config.py::get()` | `get('section.key', default=X)` |
| Config write + persist | `config.py::set_value(), save()` | Writes settings.json |
| ENVI load | `services/data_loader.py` | Returns `(cube: ndarray H,W,B, waves: list)` |
| Mode conversion | `services/processing_service.py::convert_to_ref()` | Raw→Ref; `convert_to_ref_flat()` for pixel arrays |
| Full preprocessing | `services/processing_service.py::process_cube()` | Single-entry for all chain ops |
| Background masking | `models/processing.py::create_background_mask()` | Always call on Raw cube |
| SPA band selection | `services/band_selection_service.py::select_best_bands()` | Returns `(bands, scores, mean_spectrum)` |
| Model train + export | `services/learning_service.py` | `train_model()` → `export_model()` → model.json |
| Auto-ML loop | `services/optimization_service.py` | Grid-searches Gap, NDI threshold, band count |
| Async train | `services/training_worker.py::TrainingWorker` | Emits `base_data_ready` per file for L3 cache |
| Async optimize | `services/optimization_worker.py::OptimizationWorker` | Same cache protocol as TrainingWorker |
| Root MVVM state | `viewmodels/main_vm.py::MainViewModel` | file_groups, data_cache, refs, processing_mode |
| Analysis state | `viewmodels/analysis_vm.py::AnalysisViewModel` | prep_chain, threshold, 3-level cache |
| Training state | `viewmodels/training_vm.py::TrainingViewModel` | Worker orchestration, cached_base_data |
| Project save/load | `views/main_window.py` | JSON project file; debounced auto-save on signal changes |

## VM DEPENDENCY CHAIN

```
MainViewModel          # Root; injected into all others
  └── AnalysisViewModel(main_vm)
        └── TrainingViewModel(main_vm, analysis_vm)
```

Wiring happens in `MainWindow.__init__` — never instantiate VMs elsewhere.

## CONVENTIONS

- **Config:** `from config import get, set_value, save` — never read settings.json directly
- **Threading:** >100ms work → `*_worker.py`. Never in ViewModel or View.
- **ProcessingService is single source:** Never call preprocessing functions from processing.py directly in workers/VMs — always go through `ProcessingService`
- **Masking on Raw:** `create_background_mask` always receives the raw cube, not ref/abs
- **Strict Mode:** `ValueError` on insufficient bands — no silent partial results
- **AI tags:** `/// <ai>AI가 작성함</ai>` on docstrings; `# AI가 수정함:` on inline changes
- **`Ignore` group is reserved:** files in it excluded from training labels automatically

## ANTI-PATTERNS

- Services importing from `views` or `viewmodels` — dependency inversion violation
- Running `LearningService`, `SPA`, or `process_cube()` on main thread
- Calling `apply_snv()` in any production pipeline — MROI-incompatible (needs full-frame stats)
- Duplicating preprocessing logic outside `ProcessingService`
- Committing `venv/` directory
- Modifying `models/processing.py` without verifying C# runtime parity
