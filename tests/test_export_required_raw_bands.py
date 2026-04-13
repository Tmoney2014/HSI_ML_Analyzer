"""
Tests for RequiredRawBands computation in export_model().

Algorithm (current contract):
  selected_bands are indices in the PROCESSED feature space used to train the model.
  export_model() maps each processed feature index back to the set of RAW sensor
  bands required to reproduce that feature after preprocessing.

  This matters when preprocessing changes positional mapping (e.g. SimpleDeriv).
  Example:
    total_bands=10, SimpleDeriv(gap=5, order=1), SG(win=3,radius=1)
    processed feature count = 5
    processed index 4 depends on raw features at pre-SG positions 4 and 9,
    which expand to raw neighbourhoods {3,4,5} and {8,9} after clamping.
    => RequiredRawBands = [3,4,5,8,9]
"""
import json
import sys
import pytest
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Analysis"))
from services.learning_service import LearningService


def _service():
    return LearningService.__new__(LearningService)


def _binary_svc(n_features):
    model = LinearSVC()
    model.coef_ = np.ones((1, n_features), dtype=float)
    model.intercept_ = np.array([0.0], dtype=float)
    model.classes_ = np.array([0, 1])
    return model


def call_export(selected_bands, preprocessing_config, tmp_path, total_bands=200):
    # coef_ shape must match len(original_bands) before dedup inside export_model.
    # export_model does: original_bands = list(selected_bands); selected_bands = sorted(set(...))
    # band_to_col enumerates original_bands, so col indices go 0..len(original_bands)-1.
    n_features = len(selected_bands)
    svc = _service()
    model = _binary_svc(n_features)
    out = str(tmp_path / "model.json")
    svc.export_model(
        model, selected_bands, out,
        preprocessing_config=preprocessing_config,
        total_bands=total_bands,
    )
    return json.load(open(out, encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test 1 – no preprocessing
# ---------------------------------------------------------------------------

def test_no_preprocessing(tmp_path):
    """No deriv, no SG → RequiredRawBands == SelectedBands."""
    data = call_export(
        selected_bands=[10, 20],
        preprocessing_config=[],
        tmp_path=tmp_path,
        total_bands=200,
    )
    assert data["RequiredRawBands"] == [10, 20]


# ---------------------------------------------------------------------------
# Test 2 – SimpleDeriv only (gap=5, order=1)
# ---------------------------------------------------------------------------

def test_simple_deriv_only(tmp_path):
    """Gap=5, Order=1, selected=[10,20] → base_bands={10,15,20,25}."""
    data = call_export(
        selected_bands=[10, 20],
        preprocessing_config=[{"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}}],
        tmp_path=tmp_path,
        total_bands=200,
    )
    assert data["RequiredRawBands"] == [10, 15, 20, 25]


# ---------------------------------------------------------------------------
# Test 3 – SimpleDeriv order=2 (gap=3)
# ---------------------------------------------------------------------------

def test_simple_deriv_order2(tmp_path):
    """Gap=3, Order=2, selected=[10] → base_bands={10,13,16}."""
    data = call_export(
        selected_bands=[10],
        preprocessing_config=[{"name": "SimpleDeriv", "params": {"gap": 3, "order": 2}}],
        tmp_path=tmp_path,
        total_bands=200,
    )
    assert data["RequiredRawBands"] == [10, 13, 16]


# ---------------------------------------------------------------------------
# Test 4 – SG only (win=5 → radius=2)
# ---------------------------------------------------------------------------

def test_sg_only(tmp_path):
    """SGWin=5 (radius=2), selected=[10,50] → expanded neighbourhoods."""
    data = call_export(
        selected_bands=[10, 50],
        preprocessing_config=[{"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}}],
        tmp_path=tmp_path,
        total_bands=200,
    )
    assert data["RequiredRawBands"] == [8, 9, 10, 11, 12, 48, 49, 50, 51, 52]


# ---------------------------------------------------------------------------
# Test 5 – SG clamp at zero
# ---------------------------------------------------------------------------

def test_sg_clamp_zero(tmp_path):
    """SGWin=5 (radius=2), selected=[1] → no negative indices."""
    data = call_export(
        selected_bands=[1],
        preprocessing_config=[{"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}}],
        tmp_path=tmp_path,
        total_bands=200,
    )
    assert all(b >= 0 for b in data["RequiredRawBands"]), "No negative band indices allowed"
    assert data["RequiredRawBands"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Test 6 – Deriv + SG combined
# ---------------------------------------------------------------------------

def test_deriv_and_sg_combined(tmp_path):
    """Gap=5, Order=1, SGWin=5 (radius=2), selected=[10].
    After deriv: base={10,15}. After SG expand: {8..12} ∪ {13..17} = {8..17}.
    """
    config = [
        {"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}},
        {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
    ]
    data = call_export(
        selected_bands=[10],
        preprocessing_config=config,
        tmp_path=tmp_path,
        total_bands=200,
    )
    assert data["RequiredRawBands"] == list(range(8, 18))


# ---------------------------------------------------------------------------
# Test 7 – clamp to total_bands
# ---------------------------------------------------------------------------

def test_clamp_to_total_bands(tmp_path):
    """total_bands=10: processed index 4 is the last valid feature.
    config: Deriv(gap=5,order=1) + SG(win=3 → radius=1), selected_processed=[4].
    Dependencies follow the actual chain order:
      SG first, then Deriv.
      processed 4 <- derivative over SG-expanded supports at positions 4 and 9
      => SG(4)={3,4,5}, SG(9)={8,9} after clamp to total_bands=10
      => union [3,4,8,9]
    """
    config = [
        {"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}},
        {"name": "SG", "params": {"win": 3, "poly": 2, "deriv": 0}},
    ]
    data = call_export(
        selected_bands=[4],
        preprocessing_config=config,
        tmp_path=tmp_path,
        total_bands=10,
    )
    assert data["RequiredRawBands"] == [3, 4, 8, 9]
    # Verify all are strictly less than total_bands
    assert all(b < 10 for b in data["RequiredRawBands"])


def test_clamp_to_total_bands_all_pass(tmp_path):
    """Same config but total_bands=12: full chain-order dependency survives.
    SG first then Deriv on processed index 4 => {3,4,5} ∪ {8,9,10}
    => [3,4,5,8,9,10]
    """
    config = [
        {"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}},
        {"name": "SG", "params": {"win": 3, "poly": 2, "deriv": 0}},
    ]
    data = call_export(
        selected_bands=[4],
        preprocessing_config=config,
        tmp_path=tmp_path,
        total_bands=12,
    )
    assert data["RequiredRawBands"] == [3, 4, 5, 8, 9, 10]


def test_processed_index_mapping_with_deriv_and_sg(tmp_path):
    """Processed feature index must map back to RAW dependencies, not be treated as RAW directly.

    total_bands=20, config: SG(win=5 → radius=2) then Deriv(gap=5,order=1)
    processed index 0 depends on SG-expanded raw supports at positions 0 and 5
      - SG(0)={0,1,2}
      - SG(5)={3,4,5,6,7}
    """
    config = [
        {"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}},
        {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
    ]
    data = call_export(
        selected_bands=[0],
        preprocessing_config=config,
        tmp_path=tmp_path,
        total_bands=20,
    )
    assert data["RequiredRawBands"] == [0, 1, 2, 5, 6, 7]


# ---------------------------------------------------------------------------
# Test 8 – total_bands=None raises ValueError
# ---------------------------------------------------------------------------

def test_total_bands_none_raises_valueerror(tmp_path):
    """total_bands=None must raise ValueError."""
    with pytest.raises(ValueError):
        call_export(
            selected_bands=[10, 20],
            preprocessing_config=[],
            tmp_path=tmp_path,
            total_bands=None,
        )


# ---------------------------------------------------------------------------
# Test 9 – output sorted, no duplicates
# ---------------------------------------------------------------------------

def test_output_sorted_no_duplicates(tmp_path):
    """Duplicate selected bands and overlapping SG windows produce sorted unique list."""
    config = [
        {"name": "SimpleDeriv", "params": {"gap": 5, "order": 1}},
        {"name": "SG", "params": {"win": 5, "poly": 2, "deriv": 0}},
    ]
    data = call_export(
        selected_bands=[10, 10, 20],
        preprocessing_config=config,
        tmp_path=tmp_path,
        total_bands=200,
    )
    bands = data["RequiredRawBands"]
    assert bands == sorted(bands), "RequiredRawBands must be sorted"
    assert len(bands) == len(set(bands)), "RequiredRawBands must have no duplicates"
