"""
Pytest fixtures for export_model() regression tests.
All fixtures use SimpleNamespace or numpy arrays - NO real model.fit() calls.
"""
import pytest
import numpy as np
from types import SimpleNamespace


N_FEATURES = 5  # number of selected bands used in tests


@pytest.fixture
def mock_binary_lda():
    """LDA binary: coef_ shape (1, N_FEATURES), intercept_ shape (1,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])  # (1, 5)
    m.intercept_ = np.array([-0.1])                   # (1,)
    m.classes_ = np.array([0, 1])
    return m


@pytest.fixture
def mock_multiclass_lda():
    """LDA multiclass: coef_ shape (3, N_FEATURES), intercept_ shape (3,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.5, 0.6],
    ])  # (3, 5)
    m.intercept_ = np.array([-0.1, -0.2, -0.3])  # (3,)
    m.classes_ = np.array([0, 1, 2])
    return m


@pytest.fixture
def mock_binary_svc():
    """LinearSVC binary: coef_ shape (1, N_FEATURES), intercept_ shape (1,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([[1.0, -1.0, 0.5, -0.5, 0.2]])  # (1, 5)
    m.intercept_ = np.array([0.0])                       # (1,)
    m.classes_ = np.array([0, 1])
    return m


@pytest.fixture
def mock_multiclass_svc():
    """LinearSVC multiclass: coef_ shape (3, N_FEATURES), intercept_ shape (3,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([
        [1.0, -1.0, 0.5, -0.5, 0.2],
        [-1.0, 1.0, -0.5, 0.5, -0.2],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ])  # (3, 5)
    m.intercept_ = np.array([0.1, -0.1, 0.0])  # (3,)
    m.classes_ = np.array([0, 1, 2])
    return m


@pytest.fixture
def mock_binary_ridge():
    """RidgeClassifier binary: coef_ shape (1, N_FEATURES), intercept_ shape (1,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([[0.3, 0.1, -0.2, 0.4, -0.1]])  # (1, 5)
    m.intercept_ = np.array([0.05])                      # (1,)
    m.classes_ = np.array([0, 1])
    return m


@pytest.fixture
def mock_multiclass_ridge():
    """RidgeClassifier multiclass: coef_ shape (3, N_FEATURES), intercept_ shape (3,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([
        [0.3, 0.1, -0.2, 0.4, -0.1],
        [-0.3, -0.1, 0.2, -0.4, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    ])  # (3, 5)
    m.intercept_ = np.array([0.05, -0.05, 0.0])  # (3,)
    m.classes_ = np.array([0, 1, 2])
    return m


@pytest.fixture
def mock_binary_logreg():
    """LogisticRegression binary: coef_ shape (1, N_FEATURES), intercept_ shape (1,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([[0.6, -0.3, 0.1, 0.2, -0.5]])  # (1, 5)
    m.intercept_ = np.array([0.01])                      # (1,)
    m.classes_ = np.array([0, 1])
    return m


@pytest.fixture
def mock_multiclass_logreg():
    """LogisticRegression multiclass: coef_ shape (3, N_FEATURES), intercept_ shape (3,)"""
    m = SimpleNamespace()
    m.coef_ = np.array([
        [0.6, -0.3, 0.1, 0.2, -0.5],
        [-0.6, 0.3, -0.1, -0.2, 0.5],
        [0.2, 0.1, 0.3, -0.1, 0.0],
    ])  # (3, 5)
    m.intercept_ = np.array([0.01, -0.01, 0.0])  # (3,)
    m.classes_ = np.array([0, 1, 2])
    return m


@pytest.fixture
def mock_binary_pls():
    """
    PLSRegression binary: uses export_coef_ and export_intercept_ (monkey-patched by _train_pls).
    export_coef_.shape = (N_FEATURES, 1), export_intercept_.shape = (1,)

    NOTE: This fixture uses features-major orientation (N_FEATURES, n_classes) = (5, 1).
    The production _train_pls() monkey-patches export_coef_ in class-major
    (n_classes, N_FEATURES) orientation. These fixtures are intentionally NOT
    used by test_export_model_shapes.py, which defines its own class-major
    factories. If you add a new test using these fixtures, verify the orientation
    matches your expectation before asserting on Weights shape.
    """
    m = SimpleNamespace()
    # PLS stores coef_ differently; exporter uses export_coef_ / export_intercept_
    m.export_coef_ = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])  # (5, 1)
    m.export_intercept_ = np.array([0.0])                             # (1,)
    m.classes_ = np.array([0, 1])
    return m


@pytest.fixture
def mock_multiclass_pls():
    """
    PLSRegression multiclass: export_coef_.shape = (N_FEATURES, 3), export_intercept_.shape = (3,)

    NOTE: This fixture uses features-major orientation (N_FEATURES, n_classes) = (5, 3).
    The production _train_pls() monkey-patches export_coef_ in class-major
    (n_classes, N_FEATURES) orientation. These fixtures are intentionally NOT
    used by test_export_model_shapes.py, which defines its own class-major
    factories. If you add a new test using these fixtures, verify the orientation
    matches your expectation before asserting on Weights shape.
    """
    m = SimpleNamespace()
    m.export_coef_ = np.array([
        [0.1, 0.5, 0.2],
        [0.2, 0.4, 0.3],
        [0.3, 0.3, 0.4],
        [0.4, 0.2, 0.5],
        [0.5, 0.1, 0.6],
    ])  # (5, 3)
    m.export_intercept_ = np.array([-0.1, -0.2, -0.3])  # (3,)
    m.classes_ = np.array([0, 1, 2])
    return m


@pytest.fixture
def base_preprocessing_config():
    """Standard preprocessing config: SG → SimpleDeriv → MinMax (in that order)."""
    return [
        {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
        {"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}},
        {"name": "MinMax", "params": {}},
    ]


@pytest.fixture
def no_preprocessing_config():
    """Empty preprocessing config (no steps)."""
    return []
