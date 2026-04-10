import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Analysis"))

from services.learning_service import LearningService


def _make_lda_binary(n_features):
    m = LinearDiscriminantAnalysis()
    m.coef_ = np.ones((1, n_features), dtype=float)
    m.intercept_ = np.array([-0.1], dtype=float)
    m.classes_ = np.array([0, 1])
    return m


def _export(model, tmp_path, preprocessing_config, selected_bands=None):
    svc = LearningService.__new__(LearningService)
    bands = selected_bands or [10, 20, 30, 40, 50]
    out = str(tmp_path / "m.json")
    svc.export_model(
        model=model,
        selected_bands=bands,
        output_path=out,
        total_bands=200,
        preprocessing_config=preprocessing_config,
    )
    return json.loads(Path(out).read_text())


def test_fps_formula_single_band(tmp_path):
    model = _make_lda_binary(1)
    data = _export(model, tmp_path, [], selected_bands=[50])

    assert data["EstimatedFPS"] == 91796.88


def test_fps_formula_10_bands(tmp_path):
    model = _make_lda_binary(10)
    bands = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data = _export(model, tmp_path, [], selected_bands=bands)

    assert data["EstimatedFPS"] == 9179.69


def test_fps_is_float(tmp_path):
    model = _make_lda_binary(1)
    data = _export(model, tmp_path, [], selected_bands=[50])

    assert isinstance(data["EstimatedFPS"], float)


def test_fps_decreases_with_more_bands(tmp_path):
    one_band_model = _make_lda_binary(1)
    five_band_model = _make_lda_binary(5)

    fps_1_band = _export(one_band_model, tmp_path, [], selected_bands=[50])["EstimatedFPS"]
    fps_5_bands = _export(five_band_model, tmp_path, [], selected_bands=[10, 20, 30, 40, 50])["EstimatedFPS"]

    assert fps_1_band > fps_5_bands
