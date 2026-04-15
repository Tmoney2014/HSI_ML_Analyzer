"""
Tests for OptimizationService 3D search space expansion.
Tests cover: band_methods × n_bands × gap outer loop.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np
import csv
import tempfile
import pathlib

# Patch PyQt5 before importing service so tests run without display
from unittest.mock import MagicMock
import sys

# Stub out PyQt5 signals so OptimizationService can be instantiated headlessly
pyqt5_mock = MagicMock()
pyqtSignal_stub = lambda *args, **kwargs: MagicMock()
pyqt5_mock.QtCore.QObject = object
pyqt5_mock.QtCore.pyqtSignal = pyqtSignal_stub
sys.modules.setdefault('PyQt5', pyqt5_mock)
sys.modules.setdefault('PyQt5.QtCore', pyqt5_mock.QtCore)


def _make_service():
    """Create an OptimizationService with a no-op log_message."""
    from services.optimization_service import OptimizationService
    svc = OptimizationService.__new__(OptimizationService)
    svc.log_message = MagicMock()
    svc.log_message.emit = MagicMock()
    return svc


def _make_params(method='spa', n_features=10, with_simpledv=True):
    """Build minimal initial_params dict."""
    prep = []
    if with_simpledv:
        prep.append({'name': 'SimpleDeriv', 'params': {'gap': 1, 'order': 1}})
    return {
        'band_selection_method': method,
        'n_features': n_features,
        'prep': prep,
    }


def test_3d_trial_count():
    """band_methods=['spa','anova_f'] × 8 n_bands × 40 gaps = 640 trials."""
    svc = _make_service()
    params = _make_params(method='spa', with_simpledv=True)

    call_count = 0
    def trial_callback(p):
        nonlocal call_count
        call_count += 1
        return 0.5

    best, history = svc.run_optimization(
        params, trial_callback, band_methods=['spa', 'anova_f']
    )

    # 2 methods × 8 n_bands (5,10,15,20,25,30,35,40) × 40 gaps = 640
    assert call_count == 640, f"Expected 640 trials, got {call_count}"
    assert len(history) == 640


def test_full_method_single_trial():
    """method='full' with SimpleDeriv → 1 n_bands × 40 gaps = 40 trials."""
    svc = _make_service()
    params = _make_params(method='full', n_features=7, with_simpledv=True)

    call_count = 0
    def trial_callback(p):
        nonlocal call_count
        call_count += 1
        return 0.6

    best, history = svc.run_optimization(
        params, trial_callback, band_methods=['full']
    )

    # full collapses band search to single value → 1 × 40 = 40 trials
    assert call_count == 40, f"Expected 40 trials for full method, got {call_count}"
    # All trials should have n_features == 7 (from initial_params)
    for p, acc in history:
        assert p['n_features'] == 7


def test_stop_propagation_all_3_levels():
    """Stop exception raised at trial N propagates through all 3 loops immediately."""
    svc = _make_service()
    params = _make_params(method='spa', with_simpledv=True)

    call_count = 0
    STOP_AT = 5  # Stop after 5th call

    def trial_callback(p):
        nonlocal call_count
        call_count += 1
        if call_count >= STOP_AT:
            raise Exception("Optimization Stopped")
        return 0.5

    # Must NOT raise — service catches the stop exception and returns cleanly
    best, history = svc.run_optimization(
        params, trial_callback, band_methods=['spa', 'anova_f']
    )

    # Stopped at trial STOP_AT; call_count == STOP_AT, history has STOP_AT-1 successful entries
    assert call_count == STOP_AT
    assert len(history) == STOP_AT - 1  # Last call raised before append


def test_report_gap_column_from_prep_chain():
    """_generate_report extracts gap from prep chain and emits Gap Size line."""
    svc = _make_service()

    logged = []
    svc.log_message.emit = lambda msg: logged.append(msg)

    # Construct a minimal history with a non-full spa trial
    params_no_gap = _make_params(method='spa', with_simpledv=False)
    params_no_gap['n_features'] = 10
    history = [(params_no_gap, 0.75)]

    # Call _generate_report directly (worker calls it explicitly now)
    svc._generate_report(params_no_gap, 0.75, history)

    # Check that report was emitted and contains Gap Size line
    full_log = '\n'.join(logged)
    assert 'Gap Size' in full_log, f"Expected 'Gap Size' in report, got:\n{full_log}"


def test_band_method_in_history_csv():
    """Each history entry carries band_selection_method from the outer loop."""
    svc = _make_service()
    params = _make_params(method='spa', with_simpledv=False)

    def trial_callback(p):
        return 0.8

    best, history = svc.run_optimization(
        params, trial_callback, band_methods=['spa', 'anova_f']
    )

    methods_in_history = {p.get('band_selection_method') for p, acc in history}
    assert 'spa' in methods_in_history, "History missing 'spa' entries"
    assert 'anova_f' in methods_in_history, "History missing 'anova_f' entries"

    # Also verify CSV output if output_dir is provided
    with tempfile.TemporaryDirectory() as tmpdir:
        svc._generate_report(best, 0.8, history, output_dir=tmpdir)
        csv_path = pathlib.Path(tmpdir) / 'optimization_history.csv'
        assert csv_path.exists(), "optimization_history.csv not created"
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        band_methods_csv = {r['band_method'] for r in rows}
        assert 'spa' in band_methods_csv
        assert 'anova_f' in band_methods_csv
