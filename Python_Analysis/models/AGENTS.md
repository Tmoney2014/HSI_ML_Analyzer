# models/ — Pure Processing Functions

**Generated:** 2026-03-10 | Single file: `processing.py` (248 lines)

## OVERVIEW
Pure mathematical functions implementing spectral preprocessing. **C# runtime parity required** — every function here must produce bit-identical results to the C# WPF inference engine.

## FUNCTIONS

| Function | Signature | Output Shape | C# Parity |
|----------|-----------|--------------|-----------|
| `create_background_mask` | `(cube, threshold, band_index)` → `bool mask (H,W)` | Spatial mask | N/A |
| `apply_mask` | `(cube, mask)` → `ndarray (N,B)` | Flattened pixels | N/A |
| `apply_absorbance` | `(data, epsilon)` → `ndarray` | Same as input | ✅ `-log10(R)` |
| `apply_simple_derivative` | `(data, gap, order, apply_ratio, ndi_threshold)` → `ndarray (N, B-gap*order)` | Fewer bands | ✅ Gap-Diff |
| `apply_rolling_3point_depth` | `(data, gap)` → `ndarray (N, B-2*gap)` | Fewer bands | ✅ Continuum Removal Lite |
| `apply_savgol` | `(data, window_size, poly_order, deriv)` → `ndarray` | Same as input | ✅ SG filter |
| `apply_snv` | `(data)` → `ndarray` | Same as input | ❌ MROI-incompatible |
| `apply_mean_centering` | `(data)` → `ndarray` | Same as input | — |
| `apply_min_subtraction` | `(data)` → `ndarray` | Same as input | — |
| `apply_l2_norm` | `(data)` → `ndarray` | Same as input | — |
| `apply_minmax_norm` | `(data)` → `ndarray` | Same as input | — |

## KEY BEHAVIORS

**`create_background_mask` modes:**
1. Complex string rules: `"b10 > 500 & b20 < 1000"` — parsed via regex + `eval`
2. Single band index (int or digit string): threshold on that band
3. Default: threshold on mean across all bands

**`apply_simple_derivative` (Gap-Diff):**
- Formula: `B[i+gap] - B[i]` (right minus left)
- With `apply_ratio=True`: NDI = `(B-A) / (B+A+ε)`
- Band count shrinks by `gap × order` per call

**`apply_rolling_3point_depth`:**
- Formula: `1 - (C / ((L+R)/2))`  where L/R are gap-spaced shoulders
- Band count shrinks by `2 × gap`

## CONVENTIONS

- **Pure functions only** — no side effects, no I/O, no state
- **Strict Mode enforced:** Both derivative functions raise `ValueError` on insufficient band count — never return partial results
- **Input shape:** `(N_pixels, Bands)` for all processing functions; `(H, W, B)` only for masking
- **Do not call directly from workers/VMs** — route through `ProcessingService` for consistency

## ANTI-PATTERNS

- Do NOT add stateful classes here
- Do NOT call `apply_snv()` in any training or inference pipeline
- Do NOT modify formulas without verifying against C# runtime implementation
- Do NOT import PyQt5 or any UI framework here
