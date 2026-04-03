# models/ — Pure Processing Functions

**Generated:** 2026-04-03 | Single file: `processing.py` (203 lines) <!-- AI가 수정함: LOC 250→203 (실측값), 날짜 갱신 -->

## OVERVIEW
Pure mathematical functions implementing spectral preprocessing. **C# runtime parity required** — every function here must produce bit-identical results to the C# WPF inference engine.

## FUNCTIONS

| Function | Signature | Output Shape | C# Parity |
|----------|-----------|--------------|-----------|
| `create_background_mask` | `(cube, threshold, band_index)` → `bool mask (H,W)` | Spatial mask | N/A |
| `apply_mask` | `(cube, mask)` → `ndarray (N,B)` | Flattened pixels | N/A |
| `apply_absorbance` | `(data, epsilon)` → `ndarray` | Same as input | ✅ `-log10(R)` |
| `apply_simple_derivative` | `(data, gap: int=5, order: int=1)` → `ndarray (N, B-gap*order)` | Fewer bands | ✅ Gap-Diff | <!-- AI가 수정함: 실제 시그니처로 수정 — apply_ratio/ndi_threshold 파라미터 없음 -->
| ~~`apply_rolling_3point_depth`~~ | 미구현 | — | ❌ 미구현 (Python/C# 모두 없음) |
| `apply_savgol` | `(data, window_size, poly_order, deriv)` → `ndarray` | Same as input | ✅ SG filter |
| `apply_snv` | `(data)` → `ndarray` | Same as input | ⚠️ `ddof=1` parity; requires dataset-level validation for MROI |
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
- Band count shrinks by `gap × order` per call
- `apply_ratio` / NDI 로직은 `ProcessingService` 체인 단계 파라미터로 처리됨 — 이 함수 시그니처에는 없음 <!-- AI가 수정함: apply_ratio/ndi_threshold 파라미터가 모델 함수가 아닌 서비스 레이어에 있음을 명시 -->

**~~`apply_rolling_3point_depth`~~ (미구현):**
- Python `processing.py` 및 C# 런타임 모두 구현되어 있지 않음 — 문서 오류였음

## CONVENTIONS

- **Pure functions only** — no side effects, no I/O, no state
- **Strict Mode enforced:** Both derivative functions raise `ValueError` on insufficient band count — never return partial results
- **Input shape:** `(N_pixels, Bands)` for all processing functions; `(H, W, B)` only for masking
- **Do not call directly from workers/VMs** — route through `ProcessingService` for consistency

## ANTI-PATTERNS

- Do NOT add stateful classes here
- Do NOT apply `apply_snv()` without dataset-level validation (especially in MROI)
- Do NOT modify formulas without verifying against C# runtime implementation
- Do NOT import PyQt5 or any UI framework here
