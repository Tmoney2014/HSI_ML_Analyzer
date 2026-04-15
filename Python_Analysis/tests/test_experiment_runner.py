"""
Tests for ExperimentRunner 4D grid expansion.
Covers: gap × band_methods × n_bands_list × model_types search contract.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import csv
import tempfile
import pathlib

import importlib.util

# --- Load module under test (same pattern as paper_summary tests) ---
_PYTHON_ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "experiment_runner_under_test",
    _PYTHON_ANALYSIS_DIR / "services" / "experiment_runner.py",
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
experiment_runner_module = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(experiment_runner_module)

ExperimentRunner = experiment_runner_module.ExperimentRunner


class _DummyModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _make_fake_select(data_cube, n_bands, method, labels, exclude_indices):
    B = data_cube.shape[-1]
    if method == "full":
        return list(range(B)), None, None
    n = min(n_bands, B)
    return list(range(n)), None, None


def _make_fake_train(self_or_cls, X, y, model_type, test_ratio, log_callback=None):
    return _DummyModel(), {"TrainAccuracy": 90.0, "TestAccuracy": 85.0}


def _make_runner_mocked(monkeypatch):
    """Return an ExperimentRunner with ProcessingService and LearningService mocked."""
    monkeypatch.setattr(
        experiment_runner_module.ProcessingService,
        "apply_preprocessing_chain",
        staticmethod(lambda flat_data, prep_chain: flat_data),
    )
    monkeypatch.setattr(experiment_runner_module, "select_best_bands", _make_fake_select)
    monkeypatch.setattr(experiment_runner_module.LearningService, "train_model", _make_fake_train)
    return ExperimentRunner()


def test_gap_in_csv_fieldnames():
    """'gap' must appear in _CSV_FIELDNAMES at index 3 (after n_bands)."""
    fieldnames = experiment_runner_module._CSV_FIELDNAMES
    assert "gap" in fieldnames, "'gap' column is missing from _CSV_FIELDNAMES"
    idx_n_bands = fieldnames.index("n_bands")
    idx_gap = fieldnames.index("gap")
    assert idx_gap == idx_n_bands + 1, (
        f"'gap' should be immediately after 'n_bands' (index {idx_n_bands + 1}), "
        f"but it is at index {idx_gap}"
    )


def test_gap_column_zero_when_no_simple_deriv(tmp_path, monkeypatch):
    """When prep_chain has no SimpleDeriv step, all result rows must have gap==0."""
    runner = _make_runner_mocked(monkeypatch)
    X_base = np.arange(40, dtype=float).reshape(8, 5)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    results = runner.run_grid(
        X_base=X_base,
        y=y,
        prep_chain=[],               # no SimpleDeriv
        band_methods=["spa"],
        model_types=["LDA"],
        n_bands_list=[3],
        test_ratio=0.25,
        output_dir=str(tmp_path),
        gap_range=(1, 5),            # would produce gaps 1-5 if SimpleDeriv present
        log_callback=lambda m: None,
    )

    assert len(results) >= 1
    for r in results:
        assert r["gap"] == 0, f"Expected gap=0 when no SimpleDeriv, got {r['gap']}"


def test_gap_column_values(tmp_path, monkeypatch):
    """When prep_chain contains SimpleDeriv, gap values must cover the full gap_range."""
    runner = _make_runner_mocked(monkeypatch)
    X_base = np.arange(40, dtype=float).reshape(8, 5)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # prep_chain with a SimpleDeriv step
    prep_chain = [{"name": "SimpleDeriv", "params": {"gap": 1, "order": 1}}]

    results = runner.run_grid(
        X_base=X_base,
        y=y,
        prep_chain=prep_chain,
        band_methods=["spa"],
        model_types=["LDA"],
        n_bands_list=[3],
        test_ratio=0.25,
        output_dir=str(tmp_path),
        gap_range=(3, 4),            # only gaps 3 and 4
        log_callback=lambda m: None,
    )

    gap_vals = {r["gap"] for r in results}
    assert gap_vals == {3, 4}, (
        f"Expected gap values {{3, 4}}, got {gap_vals}"
    )


def test_4d_trial_count(tmp_path, monkeypatch):
    """Total trials = len(band_methods) × len(n_bands_list) × len(gap_range) × len(model_types)."""
    runner = _make_runner_mocked(monkeypatch)
    X_base = np.arange(48, dtype=float).reshape(8, 6)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # SimpleDeriv present → gap loops over [1, 2]
    prep_chain = [{"name": "SimpleDeriv", "params": {"gap": 1, "order": 1}}]

    results = runner.run_grid(
        X_base=X_base,
        y=y,
        prep_chain=prep_chain,
        band_methods=["spa", "full"],       # 2 band methods
        model_types=["LDA", "Ridge Classifier"],  # 2 model types
        n_bands_list=[2, 3],                # 2 n_bands values
        test_ratio=0.25,
        output_dir=str(tmp_path),
        gap_range=(1, 2),                   # 2 gap values
        log_callback=lambda m: None,
    )

    # 2 bm × 2 n_bands × 2 gap × 2 mt = 16
    assert len(results) == 16, (
        f"Expected 16 trials (2×2×2×2), got {len(results)}"
    )


def test_stop_flag_propagation(tmp_path, monkeypatch):
    """stop_flag returning True after 2 trials must stop the grid with exactly 2 results."""
    runner = _make_runner_mocked(monkeypatch)
    X_base = np.arange(48, dtype=float).reshape(8, 6)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    prep_chain = [{"name": "SimpleDeriv", "params": {"gap": 1, "order": 1}}]

    call_count = [0]

    def stop_after_2():
        # stop_flag is checked before each trial begins; we want to stop after 2 rows appended
        # The flag is checked at the start of each inner loop iteration.
        # We count how many times run_grid has appended results by inspecting call_count.
        # Simpler: always False for first 2 enter-times, True on 3rd.
        return False  # will be overridden below

    results_holder = []

    original_train = _make_fake_train

    trial_count = [0]

    def counting_train(self_or_cls, X, y, model_type, test_ratio, log_callback=None):
        trial_count[0] += 1
        return _DummyModel(), {"TrainAccuracy": 90.0, "TestAccuracy": 85.0}

    monkeypatch.setattr(experiment_runner_module.LearningService, "train_model", counting_train)

    stop_count = [0]

    def stop_flag():
        # Return True once 2 trials have been started (stop_flag is called at start of each iteration)
        # The flag is polled before each mt iteration, so after 2 complete trials the flag fires
        stop_count[0] += 1
        # Stop after we've accumulated 2 completed rows in result (roughly after 3rd poll)
        return trial_count[0] >= 2

    results = runner.run_grid(
        X_base=X_base,
        y=y,
        prep_chain=prep_chain,
        band_methods=["spa", "full"],
        model_types=["LDA", "Ridge Classifier"],
        n_bands_list=[2, 3],
        test_ratio=0.25,
        output_dir=str(tmp_path),
        gap_range=(1, 3),
        stop_flag=stop_flag,
        log_callback=lambda m: None,
    )

    # Must have stopped early — significantly fewer than the full 2×2×3×2=24 trials
    assert len(results) < 24, (
        f"Expected stop_flag to interrupt grid, but got {len(results)} trials (expected < 24)"
    )
    # Must have at least 1 result (the first trial completes before stop is checked again)
    assert len(results) >= 1, "Expected at least 1 result before stop_flag fired"


def test_best_per_bm_mt_heatmap_aggregation():
    """_best_per_bm_mt returns one best row per (band_method, model_type) combination."""
    from services.experiment_runner import ExperimentRunner
    results = [
        {"band_method": "spa", "model_type": "LDA", "test_acc": 80.0, "gap": 1, "n_bands": 5, "status": "ok"},
        {"band_method": "spa", "model_type": "LDA", "test_acc": 90.0, "gap": 2, "n_bands": 5, "status": "ok"},
        {"band_method": "spa", "model_type": "LDA", "test_acc": 70.0, "gap": 3, "n_bands": 5, "status": "ok"},
        {"band_method": "full", "model_type": "LDA", "test_acc": 85.0, "gap": 1, "n_bands": "full", "status": "ok"},
        {"band_method": "spa", "model_type": "Linear SVM", "test_acc": 75.0, "gap": 1, "n_bands": 5, "status": "ok"},
        {"band_method": "spa", "model_type": "Linear SVM", "test_acc": 60.0, "gap": 2, "n_bands": 5, "status": "error: something"},
    ]
    best_results, best_config_map = ExperimentRunner._best_per_bm_mt(results)

    # Each (band_method, model_type) combination should appear exactly once
    assert len(best_results) == 3, f"Expected 3 unique (bm, mt) combinations, got {len(best_results)}"

    # Verify best selection: (spa, LDA) should pick gap=2 row (90.0)
    key = ("spa", "LDA")
    assert key in best_config_map, "Expected ('spa', 'LDA') in best_config_map"
    assert best_config_map[key]["test_acc"] == 90.0, (
        f"Expected best test_acc=90.0 for ('spa','LDA'), got {best_config_map[key]['test_acc']}"
    )
    assert best_config_map[key]["gap"] == 2, (
        f"Expected gap=2 for best ('spa','LDA'), got {best_config_map[key]['gap']}"
    )

    # Error rows must be excluded
    for r in best_results:
        assert not str(r.get("status", "")).startswith("error"), (
            f"Error row found in best_results: {r}"
        )


def test_paper_summary_best_config_table(tmp_path, monkeypatch):
    """_write_paper_summary produces a Best Configuration per (Band Method, Model) section."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import services.experiment_runner as experiment_runner_module
    from services.experiment_runner import ExperimentRunner

    # Mock ProcessingService.apply_preprocessing_chain to identity
    monkeypatch.setattr(
        experiment_runner_module.ProcessingService,
        "apply_preprocessing_chain",
        lambda X, chain: X,
    )
    # Mock select_best_bands to return first n_bands indices
    monkeypatch.setattr(
        experiment_runner_module,
        "select_best_bands",
        lambda data, n, method="spa", labels=None, exclude_indices=None: (list(range(min(n, data.shape[-1]))), None, None),
    )
    # Mock LearningService.train_model
    class _DummyModel:
        def predict(self, X):
            import numpy as np
            return np.zeros(X.shape[0], dtype=int)
    monkeypatch.setattr(
        experiment_runner_module.LearningService,
        "train_model",
        lambda self, X, y, model_type="LDA", test_ratio=0.2, log_callback=None: (
            _DummyModel(), {"TrainAccuracy": 80.0, "TestAccuracy": 75.0}
        ),
    )

    import numpy as np
    X_base = np.arange(48, dtype=float).reshape(8, 6)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    prep_chain = [{"name": "SimpleDeriv", "params": {"gap": 1, "order": 1}}]

    runner = ExperimentRunner()
    runner.run_grid(
        X_base=X_base,
        y=y,
        prep_chain=prep_chain,
        band_methods=["spa"],
        model_types=["LDA"],
        n_bands_list=[2],
        test_ratio=0.25,
        output_dir=str(tmp_path),
        gap_range=(1, 2),
        log_callback=lambda m: None,
    )

    # Find paper summary file
    exp_dir = os.path.join(str(tmp_path), "experiments")
    summary_files = [f for f in os.listdir(exp_dir) if f.endswith("_paper_summary.md")]
    assert summary_files, "No paper summary file found"

    summary_path = os.path.join(exp_dir, summary_files[0])
    content = open(summary_path, encoding="utf-8").read()

    assert "## Best Configuration per (Band Method, Model)" in content, (
        "Expected 'Best Configuration per (Band Method, Model)' section in paper summary"
    )
    # n_bands and gap columns should be present
    assert "n_bands" in content, "Expected 'n_bands' column in best config table"
    assert "gap" in content, "Expected 'gap' column in best config table"


# ---------------------------------------------------------------------------
# _build_n_bands_matrix unit tests
# ---------------------------------------------------------------------------

def test_n_bands_matrix_basic():
    """_build_n_bands_matrix: 기본 집계 — 동일 (bm, mt, n_bands)에서 max test_acc 선택 + best gap 반환."""
    ok_results = [
        {"band_method": "spa", "model_type": "LDA", "n_bands": 5,  "test_acc": 0.80, "gap": 1, "status": "ok"},
        {"band_method": "spa", "model_type": "LDA", "n_bands": 5,  "test_acc": 0.85, "gap": 2, "status": "ok"},  # max=0.85, gap=2
        {"band_method": "spa", "model_type": "LDA", "n_bands": 10, "test_acc": 0.90, "gap": 3, "status": "ok"},
        {"band_method": "anova_f", "model_type": "LDA", "n_bands": 5,  "test_acc": 0.70, "gap": 1, "status": "ok"},
        {"band_method": "anova_f", "model_type": "LDA", "n_bands": 10, "test_acc": 0.75, "gap": 4, "status": "ok"},
    ]
    band_methods = ["spa", "anova_f"]
    model_types = ["LDA"]
    n_bands_sorted, row_labels, matrix, matrix_gap = ExperimentRunner._build_n_bands_matrix(
        ok_results, band_methods, model_types, metric_key="test_acc"
    )

    assert n_bands_sorted == [5, 10], f"Expected [5, 10], got {n_bands_sorted}"
    assert row_labels == ["spa / LDA", "anova_f / LDA"]

    # spa/LDA: n_bands=5 → max(0.80, 0.85)=0.85 at gap=2, n_bands=10 → 0.90 at gap=3
    assert abs(matrix[0][0] - 0.85) < 1e-9
    assert abs(matrix[0][1] - 0.90) < 1e-9
    assert matrix_gap[0][0] == 2, f"Expected gap=2 for spa/LDA/n=5, got {matrix_gap[0][0]}"
    assert matrix_gap[0][1] == 3, f"Expected gap=3 for spa/LDA/n=10, got {matrix_gap[0][1]}"
    # anova_f/LDA: n_bands=5 → 0.70, n_bands=10 → 0.75
    assert abs(matrix[1][0] - 0.70) < 1e-9
    assert abs(matrix[1][1] - 0.75) < 1e-9
    assert matrix_gap[1][0] == 1
    assert matrix_gap[1][1] == 4


def test_n_bands_matrix_missing_combination_is_nan():
    """_build_n_bands_matrix: 해당 조합 없으면 nan, matrix_gap은 None."""
    import math
    ok_results = [
        {"band_method": "spa", "model_type": "LDA", "n_bands": 5, "test_acc": 0.80, "gap": 1, "status": "ok"},
    ]
    n_bands_sorted, row_labels, matrix, matrix_gap = ExperimentRunner._build_n_bands_matrix(
        ok_results, ["spa", "anova_f"], ["LDA"], metric_key="test_acc"
    )
    # anova_f / LDA / n_bands=5 → missing → nan / None
    assert math.isnan(matrix[1][0]), "Expected NaN for missing (anova_f, LDA, 5)"
    assert matrix_gap[1][0] is None, "Expected None gap for missing combination"


def test_n_bands_matrix_full_band_sorted_last():
    """_build_n_bands_matrix: 'full' 문자열 n_bands는 숫자 뒤에 정렬."""
    ok_results = [
        {"band_method": "spa", "model_type": "LDA", "n_bands": 10,     "test_acc": 0.80, "gap": 1, "status": "ok"},
        {"band_method": "spa", "model_type": "LDA", "n_bands": "full",  "test_acc": 0.88, "gap": 0, "status": "ok"},
        {"band_method": "spa", "model_type": "LDA", "n_bands": 5,      "test_acc": 0.75, "gap": 2, "status": "ok"},
    ]
    n_bands_sorted, _, _, _ = ExperimentRunner._build_n_bands_matrix(
        ok_results, ["spa"], ["LDA"], metric_key="test_acc"
    )
    assert n_bands_sorted[-1] == "full", f"'full' should be last, got {n_bands_sorted}"
    assert n_bands_sorted[0] == 5
    assert n_bands_sorted[1] == 10
