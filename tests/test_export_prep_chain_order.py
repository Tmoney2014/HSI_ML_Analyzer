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


def test_prep_chain_order_preserved(tmp_path):
    model = _make_lda_binary(5)
    preprocessing_config = [
        {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
        {"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}},
        {"name": "MinMax", "params": {}},
    ]

    data = _export(model, tmp_path, preprocessing_config)

    assert data["PrepChainOrder"] == ["SG", "SimpleDeriv", "MinMax"]


def test_prep_chain_order_single(tmp_path):
    model = _make_lda_binary(5)
    preprocessing_config = [{"name": "SNV", "params": {}}]

    data = _export(model, tmp_path, preprocessing_config, selected_bands=[50])

    assert data["PrepChainOrder"] == ["SNV"]


def test_prep_chain_order_empty(tmp_path):
    model = _make_lda_binary(5)

    data = _export(model, tmp_path, [])

    assert data["PrepChainOrder"] == []


def test_prep_chain_order_reverse(tmp_path):
    model = _make_lda_binary(5)
    preprocessing_config = [
        {"name": "MinMax", "params": {}},
        {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
    ]

    data = _export(model, tmp_path, preprocessing_config)

    assert data["PrepChainOrder"] == ["MinMax", "SG"]
