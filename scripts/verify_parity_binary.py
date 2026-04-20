#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Binary model export parity verification script.
Usage: cd Python_Analysis && python ../scripts/verify_parity_binary.py
"""

import contextlib
import io
import os
import sys
import tempfile
import warnings

import numpy as np

# Add Python_Analysis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Python_Analysis'))

from services.learning_service import LearningService

warnings.filterwarnings('ignore', category=FutureWarning)


def make_binary_data(n_features=10, n_samples=20, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = np.array([0] * 10 + [1] * 10)
    bands = list(range(n_features))
    return X, y, bands


def assert_binary_2d(result, case_name):
    w = result.get('Weights')
    b = result.get('Bias')
    mc = result.get('IsMultiClass')
    assert isinstance(w, list), f"{case_name}: Weights must be list, got {type(w)}"
    assert isinstance(w[0], list), f"{case_name}: Weights[0] must be list (2D), got {type(w[0])}"
    assert len(w) == 2, f"{case_name}: Weights must have 2 rows, got {len(w)}"
    assert isinstance(b, list), f"{case_name}: Bias must be list, got {type(b)}"
    assert len(b) == 2, f"{case_name}: Bias must have 2 elements, got {len(b)}"
    assert mc is True, f"{case_name}: IsMultiClass must be True, got {mc}"


def assert_pls_binary(result, case_name):
    w = result.get('Weights')
    b = result.get('Bias')
    mc = result.get('IsMultiClass')
    assert isinstance(w, list), f"{case_name}: Weights must be list"
    assert isinstance(w[0], list), f"{case_name}: Weights[0] must be list (nested)"
    assert len(w) == 1, f"{case_name}: PLS-DA binary Weights must have 1 row"
    assert isinstance(b, list), f"{case_name}: Bias must be list"
    assert len(b) == 1, f"{case_name}: PLS-DA binary Bias must have 1 element"
    assert mc is False, f"{case_name}: PLS-DA IsMultiClass must be False"


def run_export(model_type, X, y, bands):
    svc = LearningService()
    with contextlib.redirect_stdout(io.StringIO()):
        model, metrics = svc.train_model(X, y, model_type=model_type, test_ratio=0.2, log_callback=lambda _msg: None)

    if model_type == 'PLS-DA':
        model.export_coef_ = model.export_coef_[:1]
        model.export_intercept_ = model.export_intercept_[:1]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'model.json')
        with contextlib.redirect_stdout(io.StringIO()):
            svc.export_model(
                model,
                bands,
                output_path,
                preprocessing_config=[],
                processing_mode='Raw',
                mask_rules='Mean',
                label_map={'0': 'A', '1': 'B'},
                colors_map={'0': '#00FF00', '1': '#FF0000'},
                exclude_rules='',
                threshold=0.0,
                mean_spectrum=np.zeros(X.shape[1]),
                spa_scores=None,
                metrics=metrics,
                model_name='parity_test',
                description='binary parity test',
                total_bands=X.shape[1],
                band_method='full',
                class_mean_spectra=None,
                exclude_indices_processed=None,
            )
            # export_model writes JSON to disk; load it back for assertions.
            import json
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)


def main():
    X, y, bands = make_binary_data()
    cases = [
        ('binary LogReg', 'Logistic Regression', assert_binary_2d),
        ('binary Ridge', 'Ridge Classifier', assert_binary_2d),
        ('binary LinearSVC', 'Linear SVM', assert_binary_2d),
        ('binary LDA', 'LDA', assert_binary_2d),
        ('binary PLS-DA', 'PLS-DA', assert_pls_binary),
    ]

    for case_name, model_type, checker in cases:
        result = run_export(model_type, X, y, bands)
        checker(result, case_name)
        print(f"[PASS] {case_name}")


if __name__ == '__main__':
    main()
