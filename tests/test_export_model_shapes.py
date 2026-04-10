"""
Regression tests for export_model() Weights/Bias shape contract.
Covers 5 model types × binary/multiclass + 1 LDA mismatch warning case.

NO real model.fit() calls — all models are real unfitted sklearn instances
with coef_/intercept_ manually injected so isinstance() passes correctly.
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

# Ensure Python_Analysis is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "Python_Analysis"))

from services.learning_service import LearningService
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.cross_decomposition import PLSRegression

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SELECTED_BANDS = [10, 20, 30, 40, 50]
N_FEATURES = 5


# ---------------------------------------------------------------------------
# Model factory helpers — real unfitted instances with injected attributes
# ---------------------------------------------------------------------------

def _lda_binary():
    m = LinearDiscriminantAnalysis()
    m.coef_ = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])  # (1, 5)
    m.intercept_ = np.array([-0.1])
    m.classes_ = np.array([0, 1])
    return m


def _lda_multiclass():
    m = LinearDiscriminantAnalysis()
    m.coef_ = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.5, 0.6],
    ])  # (3, 5)
    m.intercept_ = np.array([-0.1, -0.2, -0.3])
    m.classes_ = np.array([0, 1, 2])
    return m


def _svc_binary():
    m = LinearSVC()
    m.coef_ = np.array([[1.0, -1.0, 0.5, -0.5, 0.2]])  # (1, 5)
    m.intercept_ = np.array([0.0])
    m.classes_ = np.array([0, 1])
    return m


def _svc_multiclass():
    m = LinearSVC()
    m.coef_ = np.array([
        [1.0, -1.0, 0.5, -0.5, 0.2],
        [-1.0, 1.0, -0.5, 0.5, -0.2],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ])  # (3, 5)
    m.intercept_ = np.array([0.1, -0.1, 0.0])
    m.classes_ = np.array([0, 1, 2])
    return m


def _ridge_binary():
    m = RidgeClassifier()
    m.coef_ = np.array([[0.3, 0.1, -0.2, 0.4, -0.1]])  # (1, 5)
    m.intercept_ = np.array([0.05])
    # classes_ is a property backed by _label_binarizer; not needed by export_model for Ridge
    return m


def _ridge_multiclass():
    m = RidgeClassifier()
    m.coef_ = np.array([
        [0.3, 0.1, -0.2, 0.4, -0.1],
        [-0.3, -0.1, 0.2, -0.4, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    ])  # (3, 5)
    m.intercept_ = np.array([0.05, -0.05, 0.0])
    # classes_ is a property backed by _label_binarizer; not needed by export_model for Ridge
    return m


def _logreg_binary():
    m = LogisticRegression()
    m.coef_ = np.array([[0.6, -0.3, 0.1, 0.2, -0.5]])  # (1, 5)
    m.intercept_ = np.array([0.01])
    m.classes_ = np.array([0, 1])
    return m


def _logreg_multiclass():
    m = LogisticRegression()
    m.coef_ = np.array([
        [0.6, -0.3, 0.1, 0.2, -0.5],
        [-0.6, 0.3, -0.1, -0.2, 0.5],
        [0.2, 0.1, 0.3, -0.1, 0.0],
    ])  # (3, 5)
    m.intercept_ = np.array([0.01, -0.01, 0.0])
    m.classes_ = np.array([0, 1, 2])
    return m


def _pls_binary():
    """Binary PLS: export_coef_ shape (1, N_FEATURES) → shape[0]==1 → flat weights."""
    m = PLSRegression()
    m.export_coef_ = np.ones((1, N_FEATURES))       # shape[0] == 1 → binary path
    m.export_intercept_ = np.array([0.0])            # shape (1,)
    m.classes_ = np.array([0, 1])
    return m


def _pls_multiclass():
    """Multiclass PLS: export_coef_ shape (3, N_FEATURES) → shape[0]>1 → nested weights."""
    m = PLSRegression()
    m.export_coef_ = np.ones((3, N_FEATURES))        # shape[0] == 3 → multiclass path
    m.export_intercept_ = np.array([-0.1, -0.2, -0.3])
    m.classes_ = np.array([0, 1, 2])
    return m


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _export(model, tmp_path, filename="m.json"):
    svc = LearningService.__new__(LearningService)
    out = str(tmp_path / filename)
    svc.export_model(
        model=model,
        selected_bands=SELECTED_BANDS,
        output_path=out,
        total_bands=200,
    )
    return json.loads(Path(out).read_text())


# ---------------------------------------------------------------------------
# Shape assertion helpers
# ---------------------------------------------------------------------------

def _assert_binary_shape(data):
    weights = data["Weights"]
    bias = data["Bias"]
    assert isinstance(weights, list), "Weights must be a list"
    assert not isinstance(weights[0], list), "Binary Weights must be 1-D (flat)"
    assert len(weights) == N_FEATURES, f"Expected {N_FEATURES} weight values"
    assert isinstance(bias, float), f"Binary Bias must be float, got {type(bias)}"
    assert data["IsMultiClass"] is False


def _assert_multiclass_shape(data, n_classes):
    weights = data["Weights"]
    bias = data["Bias"]
    assert isinstance(weights, list), "Weights must be a list"
    assert isinstance(weights[0], list), "Multiclass Weights must be 2-D (nested)"
    assert len(weights) == n_classes, f"Expected {n_classes} rows in Weights"
    assert len(weights[0]) == N_FEATURES, f"Expected {N_FEATURES} cols in Weights"
    assert isinstance(bias, list), f"Multiclass Bias must be a list, got {type(bias)}"
    assert len(bias) == n_classes
    assert data["IsMultiClass"] is True


# ===========================================================================
# 1. LDA
# ===========================================================================

def test_lda_binary_weights_shape(tmp_path):
    data = _export(_lda_binary(), tmp_path)
    _assert_binary_shape(data)
    assert data["OriginalType"] == "LinearDiscriminantAnalysis"


def test_lda_multiclass_weights_shape(tmp_path):
    data = _export(_lda_multiclass(), tmp_path)
    _assert_multiclass_shape(data, n_classes=3)
    assert data["OriginalType"] == "LinearDiscriminantAnalysis"


# ===========================================================================
# 2. LinearSVC
# ===========================================================================

def test_linearsvc_binary_weights_shape(tmp_path):
    data = _export(_svc_binary(), tmp_path)
    _assert_binary_shape(data)
    assert data["OriginalType"] == "LinearSVC"


def test_linearsvc_multiclass_weights_shape(tmp_path):
    data = _export(_svc_multiclass(), tmp_path)
    _assert_multiclass_shape(data, n_classes=3)
    assert data["OriginalType"] == "LinearSVC"


# ===========================================================================
# 3. Ridge
# ===========================================================================

def test_ridge_binary_weights_shape(tmp_path):
    data = _export(_ridge_binary(), tmp_path)
    _assert_binary_shape(data)
    assert data["OriginalType"] == "RidgeClassifier"


def test_ridge_multiclass_weights_shape(tmp_path):
    data = _export(_ridge_multiclass(), tmp_path)
    _assert_multiclass_shape(data, n_classes=3)
    assert data["OriginalType"] == "RidgeClassifier"


# ===========================================================================
# 4. LogisticRegression
# ===========================================================================

def test_logreg_binary_weights_shape(tmp_path):
    data = _export(_logreg_binary(), tmp_path)
    _assert_binary_shape(data)
    assert data["OriginalType"] == "LogisticRegression"


def test_logreg_multiclass_weights_shape(tmp_path):
    data = _export(_logreg_multiclass(), tmp_path)
    _assert_multiclass_shape(data, n_classes=3)
    assert data["OriginalType"] == "LogisticRegression"


# ===========================================================================
# 5. PLSRegression
# ===========================================================================

def test_pls_binary_weights_shape(tmp_path):
    data = _export(_pls_binary(), tmp_path)
    _assert_binary_shape(data)
    assert data["OriginalType"] == "PLSRegression"


def test_pls_multiclass_weights_shape(tmp_path):
    data = _export(_pls_multiclass(), tmp_path)
    _assert_multiclass_shape(data, n_classes=3)
    assert data["OriginalType"] == "PLSRegression"


# ===========================================================================
# 11. LDA mismatch warning (3 classes, 2 coef rows)
# ===========================================================================

def test_lda_multiclass_coef_mismatch_warning(tmp_path):
    """3 classes but 2 coef rows → RuntimeWarning emitted, no exception raised."""
    m = LinearDiscriminantAnalysis()
    m.coef_ = np.ones((2, N_FEATURES))          # 2 rows
    m.intercept_ = np.array([-0.1, -0.2])       # 2 intercepts
    m.classes_ = np.array([0, 1, 2])            # 3 classes → mismatch

    svc = LearningService.__new__(LearningService)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svc.export_model(
            model=m,
            selected_bands=SELECTED_BANDS,
            output_path=str(tmp_path / "m.json"),
            total_bands=200,
        )

    # File must be written (no exception)
    assert (tmp_path / "m.json").exists(), "export_model() must not raise on mismatch"

    # At least one RuntimeWarning about LDA coef_ mismatch
    runtime_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "mismatch" in str(w.message).lower()
    ]
    assert len(runtime_warnings) >= 1, (
        "Expected a RuntimeWarning about LDA coef_ shape mismatch; "
        f"got warnings: {[(w.category, str(w.message)) for w in caught]}"
    )
