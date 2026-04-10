import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Analysis"))

from services.learning_service import LearningService


def _service():
    return LearningService.__new__(LearningService)


def _binary_svc(weights, bias=0.0):
    model = LinearSVC()
    model.coef_ = np.array([weights], dtype=float)
    model.intercept_ = np.array([bias], dtype=float)
    model.classes_ = np.array([0, 1])
    return model


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_selected_bands_sorted_ascending(tmp_path):
    svc = _service()
    model = _binary_svc([0.1, 0.2, 0.3, 0.4])
    out = tmp_path / "model.json"

    svc.export_model(model, [99, 40, 105, 42], str(out), total_bands=200)

    data = _read_json(out)
    assert data["SelectedBands"] == [40, 42, 99, 105]


def test_selected_bands_deduped(tmp_path):
    svc = _service()
    model = _binary_svc([0.1, 0.2, 0.3, 0.4])
    out = tmp_path / "model.json"

    svc.export_model(model, [40, 40, 42, 42], str(out), total_bands=200)

    data = _read_json(out)
    assert data["SelectedBands"] == [40, 42]


def test_weights_columns_match_selected_bands(tmp_path):
    svc = _service()
    model = _binary_svc([1.25, -3.5])
    out = tmp_path / "model.json"

    svc.export_model(model, [99, 40], str(out), total_bands=200)

    data = _read_json(out)
    assert data["SelectedBands"] == [40, 99]
    assert data["Weights"] == [-3.5, 1.25]
    assert data["Bias"] == 0.0
    assert data["IsMultiClass"] is False


def test_selected_bands_single_element(tmp_path):
    svc = _service()
    model = _binary_svc([2.75])
    out = tmp_path / "model.json"

    svc.export_model(model, [42], str(out), total_bands=200)

    data = _read_json(out)
    assert data["SelectedBands"] == [42]
    assert isinstance(data["Weights"], list)
    assert data["Weights"] == [2.75]
    assert data["IsMultiClass"] is False
