"""
A. SelectedBands 정렬 + Weights 열 재정렬 regression tests.

계약:
- export_model()에 비정렬 selected_bands가 전달되어도 출력 SelectedBands는 정렬됨.
- Weights 열 순서는 정렬된 SelectedBands 순서와 일치해야 함.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC

from services.learning_service import LearningService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _binary_data(n_features: int = 10, n_samples: int = 120):
    X = _RNG.normal(size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _multiclass_data(n_features: int = 10, n_samples: int = 150, n_classes: int = 3):
    X = _RNG.normal(size=(n_samples, n_features))
    y = np.array([i % n_classes for i in range(n_samples)])
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


# ---------------------------------------------------------------------------
# Tests — A-1 / A-2 / A-3
# ---------------------------------------------------------------------------


class TestSelectedBandsSorted:
    """SelectedBands must be sorted in the exported JSON regardless of input order."""

    def test_unsorted_bands_are_sorted_in_json(self):
        """비정렬 입력 → SelectedBands 정렬 출력."""
        n_features = 5
        X, y = _binary_data(n_features)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)

        original_order = [4, 1, 3, 0, 2]  # unsorted
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, original_order, total_bands=20, tmpdir=tmpdir)

        assert data["SelectedBands"] == sorted(original_order)

    def test_already_sorted_bands_unchanged(self):
        """이미 정렬된 입력 → 동일하게 출력."""
        n_features = 5
        X, y = _binary_data(n_features)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)

        sorted_order = [0, 1, 2, 3, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, sorted_order, total_bands=20, tmpdir=tmpdir)

        assert data["SelectedBands"] == sorted_order


class TestWeightColumnsReorderedWithSortedBands:
    """Weights column order must correspond to the sorted SelectedBands order."""

    def test_multiclass_weights_reordered(self):
        """Multiclass SVM: weights 열이 정렬된 밴드 순서로 재배치됨."""
        n_features = 5
        X, y = _multiclass_data(n_features, n_classes=3)
        model = LinearSVC(dual=False, max_iter=3000).fit(X, y)

        original_order = [4, 1, 3, 0, 2]
        sorted_order = sorted(original_order)

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, original_order, total_bands=20, tmpdir=tmpdir)

        weights = np.array(data["Weights"])  # (n_classes, n_features)

        # expected: model.coef_ columns reordered from original_order → sorted_order
        band_to_col = {b: i for i, b in enumerate(original_order)}
        expected_col_order = [band_to_col[b] for b in sorted_order]
        expected_weights = model.coef_[:, expected_col_order]

        np.testing.assert_allclose(weights, expected_weights)

    def test_binary_weights_reordered(self):
        """Binary SVM: parity-fix1 이후 2D weights[1] (+w row)가 정렬된 밴드 순서로 재배치됨.
        AI가 수정함: binary 2D 계약 반영 (2026-04-21)
        """
        n_features = 5
        X, y = _binary_data(n_features)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)

        original_order = [4, 1, 3, 0, 2]
        sorted_order = sorted(original_order)

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, original_order, total_bands=20, tmpdir=tmpdir)

        weights = np.array(data["Weights"])  # (2, n_features) for binary

        band_to_col = {b: i for i, b in enumerate(original_order)}
        expected_col_order = [band_to_col[b] for b in sorted_order]
        # weights[1] = +raw_coef reordered, weights[0] = -raw_coef reordered
        expected_pos_row = model.coef_[0][expected_col_order]

        assert weights.shape == (2, n_features), f"Binary weights must be (2, n_features), got {weights.shape}"
        np.testing.assert_allclose(weights[1], expected_pos_row)
        np.testing.assert_allclose(weights[0], -expected_pos_row)

    def test_lda_multiclass_weights_reordered(self):
        """Multiclass LDA: weights 열이 정렬된 밴드 순서로 재배치됨."""
        n_features = 5
        X, y = _multiclass_data(n_features, n_classes=3)
        model = LinearDiscriminantAnalysis().fit(X, y)

        original_order = [4, 1, 3, 0, 2]
        sorted_order = sorted(original_order)

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, original_order, total_bands=20, tmpdir=tmpdir)

        weights = np.array(data["Weights"])

        band_to_col = {b: i for i, b in enumerate(original_order)}
        expected_col_order = [band_to_col[b] for b in sorted_order]
        expected_weights = model.coef_[:, expected_col_order]

        np.testing.assert_allclose(weights, expected_weights)
