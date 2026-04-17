"""
F. PLS-DA affine parity regression tests.

계약: export_model()이 내보낸 Weights(n_classes × n_bands) + Bias(n_classes)를
사용하는 선형 추론(X @ W.T + b)의 argmax가 model.predict()의 argmax와
수치적으로 허용 오차 내에서 일치해야 한다.

현재 exporter는 훈련 평균(x_mean, y_mean)으로 편향을 계산하므로
sklearn 내부의 부동소수점 그룹화와 미세한 차이가 있을 수 있다.
따라서 정확한 bit-equality가 아닌 고합의율(≥ 95%)로 검증한다.

AI가 수정함: Oracle Q1 권고 사항 구현 — PLS 수치 parity 테스트 추가
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.learning_service import LearningService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _pls_data(n_features=10, n_samples=200, n_classes=3):
    """Linearly separable enough for PLS-DA to converge."""
    X = _RNG.normal(size=(n_samples, n_features))
    # Add class-specific signal so PLS can find meaningful components
    y = np.array([i % n_classes for i in range(n_samples)])
    for c in range(n_classes):
        X[y == c, :2] += (c - 1) * 2.0
    return X, y


def _binary_pls_data(n_features=8, n_samples=160):
    X = _RNG.normal(size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _train_and_export(X, y, tmpdir):
    """Train PLS-DA via LearningService and export; return (model, exported_dict)."""
    svc = LearningService()
    model, _ = svc.train_model(X, y, model_type="PLS-DA", test_ratio=0.2)
    n_features = X.shape[1]
    selected = list(range(n_features))
    out = os.path.join(tmpdir, "model.json")
    svc.export_model(
        model,
        selected_bands=selected,
        output_path=out,
        preprocessing_config=[],
        total_bands=n_features + 10,
    )
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    return model, data


# ---------------------------------------------------------------------------
# Tests — F-1  Multiclass PLS parity
# ---------------------------------------------------------------------------


class TestPlsAffineParity:

    def test_3class_argmax_agreement(self):
        """Exported W/b argmax agrees with model.predict() argmax ≥ 95% on full data."""
        X, y = _pls_data(n_features=10, n_samples=300, n_classes=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data = _train_and_export(X, y, tmpdir)

        assert data["IsMultiClass"] is True
        W = np.array(data["Weights"])   # (n_classes, n_features)
        b = np.array(data["Bias"])       # (n_classes,)
        assert W.ndim == 2
        assert b.ndim == 1

        # Exported linear inference
        raw_scores = X @ W.T + b                   # (n_samples, n_classes)
        pred_exported = np.argmax(raw_scores, axis=1)

        # Reference: model.predict() OHE scores → argmax
        model_scores = model.predict(X)             # (n_samples, n_classes)
        pred_model = np.argmax(model_scores, axis=1)

        agreement = np.mean(pred_exported == pred_model)
        assert agreement >= 0.95, (
            f"PLS 3-class affine parity too low: {agreement:.2%}. "
            "Exported W/b may not reproduce model.predict() correctly."
        )

    def test_2class_argmax_agreement(self):
        """Binary PLS: exported W/b argmax agrees with model.predict() argmax ≥ 95%."""
        X, y = _binary_pls_data(n_features=8, n_samples=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            model, data = _train_and_export(X, y, tmpdir)

        # PLS-DA always exports as multiclass even for 2 classes
        W = np.array(data["Weights"])
        b = np.array(data["Bias"])

        if data["IsMultiClass"]:
            # Multiclass path: W is 2D (n_classes, n_features)
            raw_scores = X @ W.T + b
            pred_exported = np.argmax(raw_scores, axis=1)
        else:
            # Binary path: W is 1D (n_features,)
            raw_scores = X @ np.array(W) + b
            pred_exported = (raw_scores > 0).astype(int)

        model_scores = model.predict(X)             # (n_samples, n_classes)
        pred_model = np.argmax(model_scores, axis=1)

        agreement = np.mean(pred_exported == pred_model)
        assert agreement >= 0.95, (
            f"PLS 2-class affine parity too low: {agreement:.2%}."
        )

    def test_exported_weights_shape_matches_features(self):
        """Exported Weights row width equals the number of SelectedBands."""
        X, y = _pls_data(n_features=10, n_samples=300, n_classes=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            _, data = _train_and_export(X, y, tmpdir)

        n_selected = len(data["SelectedBands"])
        W = data["Weights"]
        assert isinstance(W, list)
        # Each row must have length == n_selected
        for row in W:
            assert isinstance(row, list)
            assert len(row) == n_selected, (
                f"Weight row length {len(row)} != SelectedBands length {n_selected}"
            )

    def test_exported_bias_length_matches_weights_rows(self):
        """Exported Bias length equals number of weight rows (n_classes)."""
        X, y = _pls_data(n_features=10, n_samples=300, n_classes=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            _, data = _train_and_export(X, y, tmpdir)

        W = data["Weights"]
        b = data["Bias"]
        assert len(b) == len(W), (
            f"Bias length {len(b)} != Weights rows {len(W)}"
        )
