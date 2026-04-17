"""
C. PrepChainOrder regression tests.

계약:
- export_model()의 PrepChainOrder는 preprocessing_config 입력 순서를 그대로 보존한다.
- 빈 preprocessing_config → PrepChainOrder == [].
- 각 step의 'name' 값만 순서대로 추출한다 (params 미포함).
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


def _binary_data(n_features: int = 8, n_samples: int = 100):
    X = _RNG.normal(size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


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


def _make_model(n_features: int = 8):
    X, y = _binary_data(n_features)
    return LinearSVC(dual=False, max_iter=2000).fit(X, y)


# ---------------------------------------------------------------------------
# Tests — C-1  empty chain
# ---------------------------------------------------------------------------


class TestPrepChainOrderEmpty:
    """Empty preprocessing_config → PrepChainOrder == []."""

    def test_no_prep_chain_gives_empty_list(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir)
        assert data["PrepChainOrder"] == []

    def test_explicit_empty_list_gives_empty_list(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(
                model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=[]
            )
        assert data["PrepChainOrder"] == []


# ---------------------------------------------------------------------------
# Tests — C-2  single step
# ---------------------------------------------------------------------------


class TestPrepChainOrderSingleStep:
    """Single preprocessing step → PrepChainOrder has exactly that one name."""

    def test_sg_single_step(self):
        model = _make_model()
        chain = [{"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["SG"]

    def test_simplederiv_single_step(self):
        # SimpleDeriv reduces band count; use only 5 of 8 bands so indices stay in range.
        n_features = 5
        X, y = _binary_data(n_features)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)
        chain = [{"name": "SimpleDeriv", "params": {"gap": 1, "order": 1}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(n_features)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["SimpleDeriv"]

    def test_minmax_single_step(self):
        model = _make_model()
        chain = [{"name": "MinMax", "params": {}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["MinMax"]

    def test_l2_single_step(self):
        model = _make_model()
        chain = [{"name": "L2", "params": {}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["L2"]

    def test_minsub_single_step(self):
        model = _make_model()
        chain = [{"name": "MinSub", "params": {}}]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["MinSub"]


# ---------------------------------------------------------------------------
# Tests — C-3  multi-step order preservation
# ---------------------------------------------------------------------------


class TestPrepChainOrderMultiStep:
    """Multiple steps → PrepChainOrder preserves insertion order exactly."""

    def test_sg_then_minmax_order_preserved(self):
        model = _make_model()
        chain = [
            {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
            {"name": "MinMax", "params": {}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["SG", "MinMax"]

    def test_minmax_then_sg_order_preserved(self):
        """Reversed insertion order must appear reversed in PrepChainOrder."""
        model = _make_model()
        chain = [
            {"name": "MinMax", "params": {}},
            {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["MinMax", "SG"]

    def test_three_step_order_preserved(self):
        model = _make_model()
        chain = [
            {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
            {"name": "L2", "params": {}},
            {"name": "MinMax", "params": {}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["SG", "L2", "MinMax"]

    def test_simplederiv_then_l2_order_preserved(self):
        n_features = 5
        X, y = _binary_data(n_features)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)
        chain = [
            {"name": "SimpleDeriv", "params": {"gap": 1, "order": 1}},
            {"name": "L2", "params": {}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(n_features)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert data["PrepChainOrder"] == ["SimpleDeriv", "L2"]

    def test_prep_chain_order_length_matches_steps(self):
        model = _make_model()
        chain = [
            {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
            {"name": "MinSub", "params": {}},
            {"name": "L2", "params": {}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, list(range(8)), total_bands=20, tmpdir=tmpdir, prep_chain=chain)
        assert len(data["PrepChainOrder"]) == len(chain)
