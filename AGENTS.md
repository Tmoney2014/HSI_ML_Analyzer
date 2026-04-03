# HSI ML Analyzer - Knowledge Base

**Generated:** 2026-04-03  
**Commit:** be1e3ae  
**Branch:** main

## OVERVIEW

HyperSort offline trainer (Python/PyQt5, MVVM). Produces `model.json` consumed by separate C# WPF runtime targeting high-speed inference.

## STRUCTURE

```
HSI_ML_Analyzer/
├── AGENTS.md                # Global rules + directory index (this file)
├── docs/                    # Pipeline/runtime/cache design specs
├── Python_Analysis/         # Main app (entry, MVVM, services, UI)
├── requirements.txt         # Python dependencies
├── HSI_Analyzer.spec        # PyInstaller packaging spec
├── build/ dist/ output/     # Build and run artifacts
└── .opencode/               # Local tool state
```

## DIRECTORY INDEX

| Directory | File | Purpose |
|-----------|------|---------|
| root | `AGENTS.md` | Global constraints and navigation |
| `docs/` | `docs/AGENTS.md` | Design-spec map and invariants |
| `Python_Analysis/` | `Python_Analysis/AGENTS.md` | Application-level dev guide |
| `Python_Analysis/models/` | `models/AGENTS.md` | Pure preprocessing math + parity rules |
| `Python_Analysis/services/` | `services/AGENTS.md` | Pipeline orchestration + workers |
| `Python_Analysis/viewmodels/` | `viewmodels/AGENTS.md` | MVVM state + cache invalidation |
| `Python_Analysis/views/` | `views/AGENTS.md` | UI wiring and tab/dialog/component rules |
| `Python_Analysis/views/tabs/` | `views/tabs/AGENTS.md` | Tab widgets: data, analysis, training | <!-- AI가 수정함: 누락된 서브디렉토리 항목 추가 -->
| `Python_Analysis/views/components/` | `views/components/AGENTS.md` | ImageViewer + CustomToolbar components | <!-- AI가 수정함: 누락된 서브디렉토리 항목 추가 -->
| `Python_Analysis/views/dialogs/` | `views/dialogs/AGENTS.md` | Modal dialogs: preprocessing, welcome | <!-- AI가 수정함: 누락된 서브디렉토리 항목 추가 -->

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Launch app | `Python_Analysis/main.py` | `main()` creates `QApplication` + `MainWindow` |
| VM wiring | `views/main_window.py` | Sole VM factory/injection point |
| Start training | `viewmodels/training_vm.py::run_training` | Spawns QThread + `TrainingWorker` |
| Start optimization | `viewmodels/training_vm.py::run_optimization` | Spawns QThread + `OptimizationWorker` |
| Preprocessing pipeline | `services/processing_service.py` | Single source for mask+convert+prep |
| Model train/export | `services/learning_service.py` | Train estimator + write `model.json` |
| Runtime invariants | `docs/data_pipeline_spec.md` | Raw masking, lazy processing, parity |
| Cache invariants | `docs/cache_hierarchy_spec_ko.md` | L1/L2 ownership and invalidation |

## CONVENTIONS (GLOBAL)

- MVVM boundary: services/models never import views/viewmodels.
- Long operations (>100ms) run in `*_worker.py` on QThread.
- `ProcessingService` is preprocessing single source; avoid duplicate pipelines.
- Background masking is always evaluated on Raw DN values.
- `models/processing.py` changes require C# runtime parity verification.
- `config.py` API is canonical (`get`, `set_value`, `save`, `reload_config`).

## ANTI-PATTERNS (GLOBAL)

- Running training/SPA/preprocessing on UI thread.
- Committing `Python_Analysis/venv/`, `build/`, `dist/` artifacts as source.
- Applying `apply_snv()` without dataset-level validation in MROI pipeline.
- Editing processing formulas without parity checks.

## COMMANDS

```bash
# Setup + run (Windows)
cd Python_Analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py

# Launcher scripts
Python_Analysis\run_gui.bat
Python_Analysis\setup_env.bat

# Package executable
pyinstaller HSI_Analyzer.spec
```

## NOTES

- No CI workflow file is currently tracked (`.github/workflows` absent).
- No project test suite is currently tracked; add `tests/` + `pytest` for CI.
- Keep generated artifacts and local tool caches out of architectural docs.
