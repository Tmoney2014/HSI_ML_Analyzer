# views/components/ — Reusable Widget Components

**Generated:** 2026-04-03 <!-- AI가 수정함: 날짜 갱신, image_viewer.py LOC 296→278 -->

## OVERVIEW
Two UI components: a complex multi-panel HSI image viewer (allowed MVVM exception) and a minimal Matplotlib toolbar subclass.

## FILES

| File | Class | Base | LOC | VM |
|------|-------|------|-----|----|
| `image_viewer.py` | `ImageViewer` | `QWidget` | 278 | `AnalysisViewModel` (read-only) | <!-- AI가 수정함: LOC 296→278 (실측값) -->
| `custom_toolbar.py` | `CustomToolbar` | `NavigationToolbar2QT` | ~16 | none |

---

## ImageViewer

Standalone popup window. Displays 2D pseudocolor HSI image + per-pixel preprocessed spectrum. Opened by `TabAnalysis` on file double-click; deduplicated by path.

### Constructor
```python
ImageViewer(file_path: str, analysis_vm: AnalysisViewModel)
```
- Reads `analysis_vm.main_vm.data_cache[file_path]` for the raw cube (L1 cache, read-only).
- Pulls `vm.threshold`, `vm.mask_rules`, `vm.prep_chain` for display state.

### Key methods

| Method | Role |
|--------|------|
| `load_data()` | Calls `load_hsi_data(path)`; populates `main_vm.data_cache` if miss |
| `update_view()` | Applies `processing.create_background_mask(cube, threshold, mask_rules)` for overlay |
| `plot_spectrum_single(x, y)` | Calls `ProcessingService.process_cube(...)` → per-pixel preprocessed spectrum |
| `toggle_graph()` | Shows/hides spectrum panel |
| `clear_all_points()` | Removes point markers from image canvas |

### Signals / connects
- `btn_toggle.clicked → toggle_graph()`
- `btn_clear.clicked → clear_all_points()`
- Matplotlib: `mpl_connect` for scroll / press / release / motion (pan, zoom, point pick)

### MVVM Exception (intentional)
Directly imports service/model code:
```python
from services.data_loader import load_hsi_data
from models import processing
from services.processing_service import ProcessingService
```
**Allowed exception** for single-pixel visualization responsiveness. See `views/AGENTS.md` for rationale. Heavy pipeline ops (training, optimization) still go through VM-managed workers.

If strict separation is ever needed: add `AnalysisViewModel.get_pixel_processed_spectrum(path, x, y)` and route `plot_spectrum_single` through it.

---

## CustomToolbar

`CustomToolbar(NavigationToolbar2QT)` — trivial Matplotlib toolbar subclass. No custom signals. No VM. Safe to reuse wherever a matplotlib figure toolbar is needed.

---

## ANTI-PATTERNS

- Do NOT add training/optimization logic to `ImageViewer`
- Do NOT use `ImageViewer` as a data processing pipeline — visualization only
- Do NOT instantiate `ImageViewer` from service or model code
- Do NOT add new `models.processing` calls to `ImageViewer` — route new processing through `ProcessingService`
