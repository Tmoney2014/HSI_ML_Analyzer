import csv
import importlib.util
import json
import inspect
import sys
from pathlib import Path

import numpy as np


PYTHON_ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "Python_Analysis"
if str(PYTHON_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_ANALYSIS_DIR))

_MODULE_SPEC = importlib.util.spec_from_file_location(
    "experiment_runner_under_test",
    PYTHON_ANALYSIS_DIR / "services" / "experiment_runner.py",
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
experiment_runner_module = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(experiment_runner_module)


class _DummyModel:
    def __init__(self, predicted_label=0):
        self._predicted_label = predicted_label

    def predict(self, X):
        return np.full(len(X), self._predicted_label, dtype=int)


def test_run_grid_writes_paper_summary_artifacts(tmp_path, monkeypatch):
    runner = experiment_runner_module.ExperimentRunner()
    X_base = np.arange(48, dtype=float).reshape(8, 6)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    monkeypatch.setattr(
        experiment_runner_module.ProcessingService,
        "apply_preprocessing_chain",
        staticmethod(lambda flat_data, prep_chain: flat_data),
    )

    def fake_select_best_bands(data_cube, n_bands, method, labels, exclude_indices):
        if method == "full":
            return list(range(data_cube.shape[-1])), None, None
        return [0, 2], None, None

    monkeypatch.setattr(experiment_runner_module, "select_best_bands", fake_select_best_bands)

    metric_sequence = iter(
        [
            ("LDA", {"TrainAccuracy": 99.0, "TestAccuracy": 94.0}),
            ("Linear SVM", {"TrainAccuracy": 98.0, "TestAccuracy": 92.5}),
            ("LDA", {"TrainAccuracy": 97.5, "TestAccuracy": 95.5}),
            ("Linear SVM", {"TrainAccuracy": 96.0, "TestAccuracy": 91.0}),
        ]
    )

    def fake_train_model(self, X, y, model_type, test_ratio, log_callback=None):
        expected_model_type, metrics = next(metric_sequence)
        assert model_type == expected_model_type
        return _DummyModel(predicted_label=0), metrics

    monkeypatch.setattr(experiment_runner_module.LearningService, "train_model", fake_train_model)

    run_grid_kwargs = {"n_bands_list": [2]} if "n_bands_list" in inspect.signature(runner.run_grid).parameters else {"n_bands": 2}

    results = runner.run_grid(
        X_base=X_base,
        y=y,
        prep_chain=[],
        band_methods=["spa", "full"],
        model_types=["LDA", "Linear SVM"],
        test_ratio=0.25,
        output_dir=str(tmp_path),
        **run_grid_kwargs,
    )

    assert len(results) == 4
    assert all(row["status"] == "ok" for row in results)

    exp_dir = tmp_path / "experiments"
    csv_files = list(exp_dir.glob("*_experiment_grid.csv"))
    summary_files = list(exp_dir.glob("*_paper_summary.md"))
    heatmap_files = sorted(path.name for path in exp_dir.glob("*_paper_matrix_*.png"))

    assert len(csv_files) == 1
    assert len(summary_files) == 1

    if importlib.util.find_spec("matplotlib") is None:
        assert heatmap_files == []
    else:
        assert len(heatmap_files) == 4
        assert any(name.endswith("_paper_matrix_test_acc.png") for name in heatmap_files)
        assert any(name.endswith("_paper_matrix_f1_macro.png") for name in heatmap_files)
        assert any(name.endswith("_paper_matrix_train_time_ms.png") for name in heatmap_files)
        assert any(name.endswith("_paper_matrix_n_bands_test_acc.png") for name in heatmap_files)

    # --- CSV column assertions ---
    with open(csv_files[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4
    assert [row["band_method"] for row in rows] == ["spa", "spa", "full", "full"]
    assert [row["n_bands"] for row in rows] == ["2", "2", "full", "full"]

    # selected_bands: must be a parseable JSON list of ints for all ok rows
    for row in rows:
        bands = json.loads(row["selected_bands"])
        assert isinstance(bands, list)
        assert all(isinstance(b, int) for b in bands)

    # confusion_png_path: must be non-empty when matplotlib is available
    if importlib.util.find_spec("matplotlib") is not None:
        for row in rows:
            assert row["confusion_png_path"] != "", (
                f"Expected confusion_png_path to be non-empty for ok row: {row}"
            )
            assert Path(row["confusion_png_path"]).exists(), (
                f"Confusion PNG file missing: {row['confusion_png_path']}"
            )
        # 4 confusion PNGs should have been written
        confusion_pngs = list(exp_dir.glob("*_confusion.png"))
        assert len(confusion_pngs) == 4

    # --- Paper summary content assertions ---
    summary_text = summary_files[0].read_text(encoding="utf-8")
    assert "# Experiment Matrix Paper Summary" in summary_text
    assert "## Best Overall Configuration" in summary_text
    assert "## Matrix Tables" in summary_text
    assert "### Test Accuracy (%)" in summary_text
    assert "### Macro F1" in summary_text
    assert "### Train Time (ms)" in summary_text
    assert "| 1 | full | LDA | 95.5000 |" in summary_text

    # Confusion matrices section present when matplotlib available
    if importlib.util.find_spec("matplotlib") is not None:
        assert "## Confusion Matrices" in summary_text
        assert "_confusion.png" in summary_text


def test_run_grid_writes_failure_summary_without_heatmaps(tmp_path, monkeypatch):
    runner = experiment_runner_module.ExperimentRunner()
    X_base = np.arange(24, dtype=float).reshape(6, 4)
    y = np.array([0, 0, 0, 1, 1, 1])

    monkeypatch.setattr(
        experiment_runner_module.ProcessingService,
        "apply_preprocessing_chain",
        staticmethod(lambda flat_data, prep_chain: flat_data),
    )

    def failing_select_best_bands(data_cube, n_bands, method, labels, exclude_indices):
        raise RuntimeError("synthetic selection failure")

    monkeypatch.setattr(experiment_runner_module, "select_best_bands", failing_select_best_bands)

    run_grid_kwargs = {"n_bands_list": [2]} if "n_bands_list" in inspect.signature(runner.run_grid).parameters else {"n_bands": 2}

    results = runner.run_grid(
        X_base=X_base,
        y=y,
        prep_chain=[],
        band_methods=["spa"],
        model_types=["LDA"],
        test_ratio=0.25,
        output_dir=str(tmp_path),
        **run_grid_kwargs,
    )

    assert len(results) == 1
    assert results[0]["status"].startswith("error: synthetic selection failure")

    exp_dir = tmp_path / "experiments"
    csv_files = list(exp_dir.glob("*_experiment_grid.csv"))
    summary_files = list(exp_dir.glob("*_paper_summary.md"))
    heatmap_files = list(exp_dir.glob("*_paper_matrix_*.png"))
    confusion_files = list(exp_dir.glob("*_confusion.png"))

    assert len(csv_files) == 1
    assert len(summary_files) == 1
    assert heatmap_files == []
    assert confusion_files == []  # no confusion PNGs for failed trials

    summary_text = summary_files[0].read_text(encoding="utf-8")
    assert "## No successful trials" in summary_text
    assert "All trials failed. Check the aggregate CSV and logs for error details." in summary_text

    with open(csv_files[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["status"].startswith("error: synthetic selection failure")
    # error rows have empty selected_bands and confusion_png_path
    assert rows[0]["selected_bands"] == ""
    assert rows[0]["confusion_png_path"] == ""
