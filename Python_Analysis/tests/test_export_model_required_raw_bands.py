"""
B. RequiredRawBands 계산 regression tests.

계약:
- 전처리 없을 때: RequiredRawBands == sorted(SelectedBands).
- SimpleDeriv(gap=G): processed index i → raw {i, i+G}.
- SG(win=W): processed index i → raw {max(0,i-r)..min(last,i+r)}, r = W//2.
- 출력된 RequiredRawBands의 모든 값은 [0, total_bands) 범위 내여야 함.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.svm import LinearSVC

from services.learning_service import LearningService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(1)


def _make_binary_svc(n_features: int):
    X = _RNG.normal(size=(120, n_features))
    y = (X[:, 0] > 0).astype(int)
    return LinearSVC(dual=False, max_iter=2000).fit(X, y)


def _export(selected_bands, prep_chain, total_bands):
    n_features = len(selected_bands)
    model = _make_binary_svc(n_features)
    svc = LearningService()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "model.json")
        svc.export_model(
            model,
            selected_bands=list(selected_bands),
            output_path=out,
            preprocessing_config=prep_chain,
            total_bands=total_bands,
        )
        with open(out, encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Tests — B-1 … B-4
# ---------------------------------------------------------------------------


class TestRequiredRawBandsNoPreprocesing:
    """전처리 없으면 RequiredRawBands == sorted(SelectedBands)."""

    def test_single_band(self):
        data = _export([42], [], total_bands=100)
        assert data["RequiredRawBands"] == [42]

    def test_multiple_bands(self):
        selected = [10, 30, 50]
        data = _export(selected, [], total_bands=100)
        assert data["RequiredRawBands"] == sorted(selected)

    def test_selected_bands_already_sorted(self):
        selected = [5, 10, 20]
        data = _export(selected, [], total_bands=100)
        assert data["RequiredRawBands"] == selected


class TestRequiredRawBandsSimpleDeriv:
    """SimpleDeriv(gap=G, order=1): processed i → raw {i, i+G}."""

    def test_gap5_single_band(self):
        gap = 5
        selected = [10]
        prep = [{"name": "SimpleDeriv", "params": {"gap": gap, "order": 1}}]
        data = _export(selected, prep, total_bands=100)
        assert data["RequiredRawBands"] == sorted({10, 10 + gap})

    def test_gap5_multiple_bands(self):
        gap = 5
        selected = [10, 20]
        prep = [{"name": "SimpleDeriv", "params": {"gap": gap, "order": 1}}]
        data = _export(selected, prep, total_bands=100)
        expected = sorted({10, 10 + gap, 20, 20 + gap})
        assert data["RequiredRawBands"] == expected

    def test_gap3_order1(self):
        gap = 3
        selected = [15]
        prep = [{"name": "SimpleDeriv", "params": {"gap": gap, "order": 1}}]
        data = _export(selected, prep, total_bands=100)
        assert data["RequiredRawBands"] == sorted({15, 15 + gap})

    def test_gap5_order2(self):
        """order=2: two passes of gap=5.
        Pass1: processed index i = raw {i, i+5}.
        Pass2: processed index i (in pass1-space) = {i, i+5} ∪ {i+5, i+10} = {i, i+5, i+10}.
        So selected index 10 → raw {10, 15, 20}.
        """
        gap, order = 5, 2
        selected = [10]
        prep = [{"name": "SimpleDeriv", "params": {"gap": gap, "order": order}}]
        data = _export(selected, prep, total_bands=100)
        assert data["RequiredRawBands"] == sorted({10, 15, 20})


class TestRequiredRawBandsSavitzkyGolay:
    """SG(win=W): scipy savgol_filter default mode='interp'.

    Interior positions:   processed i → raw [i-radius .. i+radius]
    Left boundary  (i < radius):        raw [0 .. win-1]
    Right boundary (i > last-radius):   raw [last-win+1 .. last]

    This matches SciPy's mode='interp' polynomial-fit anchoring at edges.
    AI가 수정함: clipped-symmetric 계약에서 mode='interp' 실제 동작 계약으로 교체
    """

    def test_win5_interior_band(self):
        """win=5 → radius=2; interior band 50 → raw {48,49,50,51,52}."""
        win = 5
        radius = win // 2  # = 2
        selected = [50]
        prep = [{"name": "SG", "params": {"win": win, "poly": 2, "deriv": 0}}]
        data = _export(selected, prep, total_bands=100)
        expected = list(range(50 - radius, 50 + radius + 1))
        assert data["RequiredRawBands"] == expected

    def test_win5_boundary_band_zero(self):
        """band 0 at left edge: mode='interp' fits poly from raw[0..4] → full window required.

        AI가 수정함: 이전 clipped-window 계약 [0..2]는 잘못됨.
        scipy mode='interp'는 경계에서 window_size 전체 포인트(raw[0..win-1])에 의존.
        """
        win = 5
        selected = [0]
        prep = [{"name": "SG", "params": {"win": win, "poly": 2, "deriv": 0}}]
        data = _export(selected, prep, total_bands=20)
        # i=0 < radius=2 → j_start=0, j_end=min(win-1, last)=4
        assert data["RequiredRawBands"] == [0, 1, 2, 3, 4]

    def test_win5_boundary_band_last(self):
        """band 19 at right edge (total=20): mode='interp' uses raw[15..19].

        AI가 수정함: 우측 경계 테스트 추가 — right boundary는 raw[last-win+1..last]에 의존.
        """
        win = 5
        total = 20
        last = total - 1  # = 19
        selected = [last]  # processed index = last after no-shrink SG
        prep = [{"name": "SG", "params": {"win": win, "poly": 2, "deriv": 0}}]
        data = _export(selected, prep, total_bands=total)
        # i=19 > last-radius=17 → j_start=max(0,19-4)=15, j_end=19
        assert data["RequiredRawBands"] == list(range(last - win + 1, last + 1))

    def test_win7_radius3(self):
        """win=7 → radius=3; band 10 → raw {7..13}."""
        win = 7
        radius = win // 2  # = 3
        selected = [10]
        prep = [{"name": "SG", "params": {"win": win, "poly": 2, "deriv": 0}}]
        data = _export(selected, prep, total_bands=100)
        expected = list(range(10 - radius, 10 + radius + 1))
        assert data["RequiredRawBands"] == expected


class TestRequiredRawBandsBoundedByTotalBands:
    """RequiredRawBands의 모든 값은 [0, total_bands) 범위 내여야 한다."""

    def test_no_prep_all_in_range(self):
        selected = [0, 10, 99]
        data = _export(selected, [], total_bands=100)
        for b in data["RequiredRawBands"]:
            assert 0 <= b < 100, f"Out-of-range band {b}"

    def test_simple_deriv_all_in_range(self):
        gap = 5
        selected = [10, 20, 30]
        prep = [{"name": "SimpleDeriv", "params": {"gap": gap, "order": 1}}]
        data = _export(selected, prep, total_bands=100)
        for b in data["RequiredRawBands"]:
            assert 0 <= b < 100, f"Out-of-range band {b}"

    def test_sg_all_in_range(self):
        prep = [{"name": "SG", "params": {"win": 9, "poly": 3, "deriv": 0}}]
        selected = [0, 5, 95]
        data = _export(selected, prep, total_bands=100)
        for b in data["RequiredRawBands"]:
            assert 0 <= b < 100, f"Out-of-range band {b}"
