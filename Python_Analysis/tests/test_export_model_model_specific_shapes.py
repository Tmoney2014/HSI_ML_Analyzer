"""
D. Model-specific Weights / Bias / IsMultiClass shape regression tests.

계약 (learning_service.py exporter):
- binary linear models (SVC/LDA/Ridge/LogReg, coef_.shape[0] == 1):
    Weights = list[float]  (1D, length == n_selected_bands)
    Bias    = float
    IsMultiClass = False
- multiclass linear models (coef_.shape[0] > 1):
    Weights = list[list[float]]  (shape: n_classes × n_selected_bands)
    Bias    = list[float]        (length == n_classes)
    IsMultiClass = True
- PLS-DA: always multiclass representation (export_coef_.shape[0] > 1 even for 2 classes)

주의: LogisticRegression은 train_model()을 경유하면 binary case에서
      assert coef_.shape[0] == len(np.unique(y))가 실패하므로 직접 fit한다.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from services.learning_service import LearningService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(2)


def _binary_data(n_features: int = 8, n_samples: int = 120):
    X = _RNG.normal(size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _multiclass_data(n_features: int = 8, n_samples: int = 150, n_classes: int = 3):
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


def _bands(n: int):
    return list(range(n))


# ---------------------------------------------------------------------------
# Tests — D-1  LinearSVC
# ---------------------------------------------------------------------------


class TestLinearSVCShapes:

    def test_binary_weights_is_flat_list_of_floats(self):
        n = 8
        X, y = _binary_data(n)
        model = LinearSVC(dual=False, max_iter=2000).fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        assert isinstance(data["Weights"], list)
        assert all(isinstance(w, float) for w in data["Weights"])
        assert len(data["Weights"]) == n
        assert isinstance(data["Bias"], float)
        assert data["IsMultiClass"] is False

    def test_multiclass_weights_is_2d_list(self):
        n = 8
        n_classes = 3
        X, y = _multiclass_data(n, n_classes=n_classes)
        model = LinearSVC(dual=False, max_iter=3000).fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights)
        assert len(weights) == n_classes
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == n_classes
        assert data["IsMultiClass"] is True


# ---------------------------------------------------------------------------
# Tests — D-2  LDA
# ---------------------------------------------------------------------------


class TestLDAShapes:

    def test_binary_lda_is_flat_weights(self):
        # LDA binary: coef_.shape == (1, n_features) → binary branch
        n = 8
        X, y = _binary_data(n)
        model = LinearDiscriminantAnalysis().fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        assert isinstance(data["Weights"], list)
        assert all(isinstance(w, float) for w in data["Weights"])
        assert len(data["Weights"]) == n
        assert isinstance(data["Bias"], float)
        assert data["IsMultiClass"] is False

    def test_multiclass_lda_is_2d_weights(self):
        n = 8
        n_classes = 3
        X, y = _multiclass_data(n, n_classes=n_classes)
        model = LinearDiscriminantAnalysis().fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights)
        assert len(weights) == n_classes
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == n_classes
        assert data["IsMultiClass"] is True


# ---------------------------------------------------------------------------
# Tests — D-3  RidgeClassifier
# ---------------------------------------------------------------------------


class TestRidgeClassifierShapes:

    def test_binary_ridge_is_flat_weights(self):
        n = 8
        X, y = _binary_data(n)
        model = RidgeClassifier(class_weight="balanced").fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        assert isinstance(data["Weights"], list)
        assert all(isinstance(w, float) for w in data["Weights"])
        assert len(data["Weights"]) == n
        assert isinstance(data["Bias"], float)
        assert data["IsMultiClass"] is False

    def test_multiclass_ridge_is_2d_weights(self):
        n = 8
        n_classes = 3
        X, y = _multiclass_data(n, n_classes=n_classes)
        model = RidgeClassifier(class_weight="balanced").fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights)
        assert len(weights) == n_classes
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == n_classes
        assert data["IsMultiClass"] is True


# ---------------------------------------------------------------------------
# Tests — D-4  LogisticRegression
#   Note: train_model() is NOT used — its assert coef_.shape[0]==n_classes
#   fails for binary case (sklearn returns coef_.shape[0]==1 for 2-class problems).
# ---------------------------------------------------------------------------


class TestLogisticRegressionShapes:

    def test_binary_logreg_is_flat_weights(self):
        # sklearn binary LogReg: coef_.shape == (1, n_features)
        n = 8
        X, y = _binary_data(n)
        model = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, y)
        assert model.coef_.shape[0] == 1, "precondition: binary coef_ must have 1 row"
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        assert isinstance(data["Weights"], list)
        assert all(isinstance(w, float) for w in data["Weights"])
        assert len(data["Weights"]) == n
        assert isinstance(data["Bias"], float)
        assert data["IsMultiClass"] is False

    def test_multiclass_logreg_is_2d_weights(self):
        n = 8
        n_classes = 3
        X, y = _multiclass_data(n, n_classes=n_classes)
        model = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, y)
        assert model.coef_.shape[0] == n_classes, "precondition: multiclass coef_ rows == n_classes"
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights)
        assert len(weights) == n_classes
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == n_classes
        assert data["IsMultiClass"] is True


# ---------------------------------------------------------------------------
# Tests — D-5  PLS-DA (always multiclass)
# ---------------------------------------------------------------------------


class TestPLSDAShapes:

    def test_plsda_two_class_is_multiclass(self):
        """PLS-DA export은 2클래스에서도 항상 multiclass 표현이어야 한다."""
        n = 8
        n_classes = 2
        X, y = _multiclass_data(n, n_classes=n_classes)
        svc = LearningService()
        model, _ = svc.train_model(X, y, model_type="PLS-DA")
        # PLS-DA monkey-patches export_coef_ to (n_classes, n_features)
        assert model.export_coef_.shape[0] == n_classes
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights)
        assert len(weights) == n_classes
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == n_classes
        assert data["IsMultiClass"] is True

    def test_plsda_three_class_is_multiclass(self):
        n = 8
        n_classes = 3
        X, y = _multiclass_data(n, n_classes=n_classes)
        svc = LearningService()
        model, _ = svc.train_model(X, y, model_type="PLS-DA")
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights)
        assert len(weights) == n_classes
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == n_classes
        assert data["IsMultiClass"] is True

    def test_plsda_weights_length_matches_selected_bands(self):
        n = 6
        X, y = _multiclass_data(n, n_classes=3)
        svc = LearningService()
        model, _ = svc.train_model(X, y, model_type="PLS-DA")
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        assert all(len(row) == n for row in data["Weights"])
