"""
D. Model-specific Weights / Bias / IsMultiClass shape regression tests.

계약 (learning_service.py exporter, parity-fix1 이후):
- binary linear models (SVC/LDA/Ridge/LogReg):
    Weights = list[list[float]]  (2D, 2 × n_selected_bands) — [[-w...], [+w...]]
    Bias    = list[float]        (length == 2)               — [-b, +b]
    IsMultiClass = True
- multiclass linear models (n_classes > 2):
    Weights = list[list[float]]  (shape: n_classes × n_selected_bands)
    Bias    = list[float]        (length == n_classes)
    IsMultiClass = True
- PLS-DA: always multiclass representation (export_coef_.shape[0] == n_classes)

AI가 수정함: parity-fix1 binary 2D 계약으로 업데이트 (2026-04-21)
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
        # parity-fix1: binary → 2D 2-class 전개 ([[-w...], [+w...]])
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights), "Binary Weights must be 2D (list of lists)"
        assert len(weights) == 2, f"Binary must have 2 rows, got {len(weights)}"
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == 2
        assert data["IsMultiClass"] is True

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
        # LDA binary: parity-fix1 → 2D 2-class 전개
        n = 8
        X, y = _binary_data(n)
        model = LinearDiscriminantAnalysis().fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights), "Binary LDA Weights must be 2D"
        assert len(weights) == 2, f"Binary must have 2 rows, got {len(weights)}"
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == 2
        assert data["IsMultiClass"] is True

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
        # parity-fix1: binary → 2D 2-class 전개
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights), "Binary Ridge Weights must be 2D"
        assert len(weights) == 2, f"Binary must have 2 rows, got {len(weights)}"
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == 2
        assert data["IsMultiClass"] is True

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
        # parity-fix1: binary LogReg → 2D 2-class 전개 (sklearn coef_.shape[0]==1 → 2 rows)
        n = 8
        X, y = _binary_data(n)
        model = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, y)
        assert model.coef_.shape[0] == 1, "precondition: binary coef_ must have 1 row"
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights), "Binary LogReg Weights must be 2D"
        assert len(weights) == 2, f"Binary must have 2 rows, got {len(weights)}"
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == 2
        assert data["IsMultiClass"] is True

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


# ---------------------------------------------------------------------------
# Tests — D-6  Binary Logistic Regression via train_model() (end-to-end)
#
# Oracle Q4 bug: _train_logistic() previously asserted coef_.shape[0] == n_classes,
# which always failed for binary LogReg (sklearn stores 1 row, not 2).
# This test validates the full train_model() → export path succeeds for binary.
#
# AI가 추가함: Binary LogReg train_model() 엔드투엔드 성공 테스트
# ---------------------------------------------------------------------------


class TestLogisticRegressionBinaryTrainModel:

    def test_binary_train_model_succeeds_without_assertion_error(self):
        """train_model() must not raise AssertionError for binary Logistic Regression."""
        n = 8
        X, y = _binary_data(n)
        svc = LearningService()
        # Should not raise AssertionError (which was the bug: coef_.shape[0]==1 != 2)
        model, metrics = svc.train_model(X, y, model_type="Logistic Regression")
        assert model is not None
        assert "TrainAccuracy" in metrics

    def test_binary_train_model_export_produces_valid_shape(self):
        """Binary LogReg export via train_model() path must yield 2D Weights and list Bias.
        AI가 수정함: parity-fix1 binary 2D 계약으로 업데이트 (2026-04-21)
        """
        n = 8
        X, y = _binary_data(n)
        svc = LearningService()
        model, _ = svc.train_model(X, y, model_type="Logistic Regression")
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _export(model, _bands(n), total_bands=20, tmpdir=tmpdir)

        assert data["IsMultiClass"] is True
        weights = data["Weights"]
        assert isinstance(weights, list)
        assert all(isinstance(row, list) for row in weights), (
            "Binary Weights must be list[list[float]] (2D), not list[float]"
        )
        assert len(weights) == 2, f"Binary export must produce 2 rows, got {len(weights)}"
        assert all(len(row) == n for row in weights)
        assert isinstance(data["Bias"], list)
        assert len(data["Bias"]) == 2

    def test_binary_train_model_returns_accuracy_in_range(self):
        """Binary LogReg train accuracy must be in [0, 100] range."""
        n = 8
        X, y = _binary_data(n)
        svc = LearningService()
        _, metrics = svc.train_model(X, y, model_type="Logistic Regression")
        assert 0.0 <= metrics["TrainAccuracy"] <= 100.0
        assert 0.0 <= metrics["TestAccuracy"] <= 100.0
