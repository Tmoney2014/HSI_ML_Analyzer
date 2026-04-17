"""
E. EstimatedFPS regression tests.

계약 (learning_service.py exporter):
    EstimatedFPS = round(940*1e6/8 / (max(len(RequiredRawBands), 1) * 2 * 640), 2)

RequiredRawBands는 export 전 clamp(total_bands)가 적용된 최종 목록.
FPS 공식의 분모는 clamp 후 길이를 사용한다.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.svm import LinearSVC

from services.learning_service import LearningService

# ---------------------------------------------------------------------------
# FPS formula constants (must mirror learning_service.py)
# ---------------------------------------------------------------------------

_BW_BPS = 940 * 1_000_000 / 8   # 117_500_000.0 bytes/sec
_PIXEL_BYTES = 2
_FRAME_WIDTH = 640


def _expected_fps(n_required_raw: int) -> float:
    return round(_BW_BPS / (max(n_required_raw, 1) * _PIXEL_BYTES * _FRAME_WIDTH), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(3)


def _binary_data(n_features: int = 5, n_samples: int = 100):
    X = _RNG.normal(size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _make_model(n_features: int = 5):
    X, y = _binary_data(n_features)
    return LinearSVC(dual=False, max_iter=2000).fit(X, y)


def _export(model, selected_bands, total_bands, tmpdir, prep_chain=None):
    svc = LearningService()
    out = os.path.join(tmpdir, "model.json")
    svc.export_model(
        model,
        selected_bands=selected_bands,
        output_path=out,
        preprocessing_config=prep_chain or [],
        total_bands=total_bands,
    )
    with open(out, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests — E-1  formula correctness (no preprocessing)
# ---------------------------------------------------------------------------


class TestEstimatedFPSFormula:
    """Without preprocessing RequiredRawBands == SelectedBands, FPS follows formula exactly."""

    def test_single_band_fps_matches_formula(self):
        model = _make_model(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, [10], total_bands=100, tmpdir=tmpdir)
        assert data["EstimatedFPS"] == pytest.approx(_expected_fps(1), abs=0.01)

    def test_five_bands_fps_matches_formula(self):
        model = _make_model(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, [0, 1, 2, 3, 4], total_bands=50, tmpdir=tmpdir)
        assert data["EstimatedFPS"] == pytest.approx(_expected_fps(5), abs=0.01)

    def test_ten_bands_fps_matches_formula(self):
        model = _make_model(10)
        X, y = _binary_data(10)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(10)), total_bands=50, tmpdir=tmpdir)
        assert data["EstimatedFPS"] == pytest.approx(_expected_fps(10), abs=0.01)


# ---------------------------------------------------------------------------
# Tests — E-2  FPS decreases as band count grows
# ---------------------------------------------------------------------------


class TestEstimatedFPSDecreasesWithMoreBands:
    """More required raw bands → lower FPS (bandwidth is fixed)."""

    def test_fps_decreases_from_1_to_5_bands(self):
        model_1 = _make_model(1)
        model_5 = _make_model(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_1 = _export(model_1, [0], total_bands=50, tmpdir=tmpdir)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_5 = _export(model_5, list(range(5)), total_bands=50, tmpdir=tmpdir)
        assert data_1["EstimatedFPS"] > data_5["EstimatedFPS"]

    def test_fps_decreases_monotonically(self):
        counts = [1, 3, 5, 8]
        fps_values = []
        for n in counts:
            X, y = _binary_data(n)
            model = LinearSVC(dual=False, max_iter=2000).fit(X, y)
            with tempfile.TemporaryDirectory() as tmpdir:
                data = _export(model, list(range(n)), total_bands=50, tmpdir=tmpdir)
            fps_values.append(data["EstimatedFPS"])
        for i in range(len(fps_values) - 1):
            assert fps_values[i] > fps_values[i + 1], (
                f"FPS[{counts[i]}]={fps_values[i]} should be > FPS[{counts[i+1]}]={fps_values[i+1]}"
            )


# ---------------------------------------------------------------------------
# Tests — E-3  FPS is always positive
# ---------------------------------------------------------------------------


class TestEstimatedFPSPositive:
    """EstimatedFPS must be > 0 regardless of band configuration."""

    def test_fps_positive_for_single_band(self):
        model = _make_model(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, [0], total_bands=50, tmpdir=tmpdir)
        assert data["EstimatedFPS"] > 0

    def test_fps_positive_for_many_bands(self):
        n = 20
        X, y = _binary_data(n)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(n)), total_bands=100, tmpdir=tmpdir)
        assert data["EstimatedFPS"] > 0


# ---------------------------------------------------------------------------
# Tests — E-4  FPS with preprocessing (band expansion via SimpleDeriv)
# ---------------------------------------------------------------------------


class TestEstimatedFPSWithPreprocessing:
    """SimpleDeriv gap>0 expands RequiredRawBands, which lowers EstimatedFPS."""

    def test_fps_with_deriv_lower_than_without(self):
        # Without prep: 3 selected bands → 3 raw bands
        # With SimpleDeriv gap=2: each band i also needs i+2 → at least some expansion
        n = 3
        selected = [0, 1, 2]
        X, y = _binary_data(n)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_no_prep = _export(model, selected, total_bands=20, tmpdir=tmpdir)

        chain = [{"name": "SimpleDeriv", "params": {"gap": 2, "order": 1}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_with_prep = _export(model, selected, total_bands=20, tmpdir=tmpdir, prep_chain=chain)

        # expanded required raw bands → lower FPS
        assert len(data_with_prep["RequiredRawBands"]) > len(data_no_prep["RequiredRawBands"])
        assert data_with_prep["EstimatedFPS"] < data_no_prep["EstimatedFPS"]

    def test_fps_matches_formula_given_required_raw_bands(self):
        """EstimatedFPS == formula(len(RequiredRawBands)) for any config."""
        n = 3
        selected = [0, 1, 2]
        X, y = _binary_data(n)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)
        chain = [{"name": "SimpleDeriv", "params": {"gap": 3, "order": 1}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, selected, total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        n_raw = len(data["RequiredRawBands"])
        assert data["EstimatedFPS"] == pytest.approx(_expected_fps(n_raw), abs=0.01)
