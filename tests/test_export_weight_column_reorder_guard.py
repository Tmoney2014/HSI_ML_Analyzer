"""
Tests for weight-column reorder length guard in export_model().

Guard rationale:
    In export_model(), original_bands = list(selected_bands_input) is captured
    before sort+dedup. The reorder uses:
        col_order = [band_to_col[b] for b in selected_bands if b in band_to_col]
    The 'if b in band_to_col' silently drops any band in selected_bands not
    present in original_bands, producing a col_order shorter than selected_bands.
    The guard (Task 4) converts this silent length mismatch into a ValueError.

Test strategy:
    We cannot produce this mismatch via a normal export_model() call because
    original_bands is always derived from selected_bands_input. Instead we use
    unittest.mock.patch to replace the module-level 'sorted' call so that
    selected_bands (post-sort) contains an extra band absent from original_bands,
    replicating a data-corruption or API-change scenario.
"""
import json
import sys
import unittest.mock as mock
import pytest
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Analysis"))
from services.learning_service import LearningService


# Save the real sorted before any patching
_real_sorted = sorted


def _service():
    return LearningService.__new__(LearningService)


def _binary_svc(n_features):
    """Binary LinearSVC: coef_ shape (1, n_features), intercept_ shape (1,)."""
    model = LinearSVC()
    model.coef_ = np.ones((1, n_features), dtype=float)
    model.intercept_ = np.array([0.0], dtype=float)
    model.classes_ = np.array([0, 1])
    return model


def _multiclass_svc(n_features, n_classes=3):
    """Multiclass LinearSVC: coef_ shape (n_classes, n_features)."""
    model = LinearSVC()
    model.coef_ = np.ones((n_classes, n_features), dtype=float)
    model.intercept_ = np.zeros(n_classes, dtype=float)
    model.classes_ = np.arange(n_classes)
    return model


def _make_patched_sorted():
    """
    Returns a patched sorted() that appends band 999 on the first set-sort call
    (the selected_bands normalisation inside export_model). 999 is NOT in
    original_bands=[10,20,30], so col_order will be shorter → guard fires.
    """
    call_count = [0]

    def patched(iterable, *args, **kwargs):
        call_count[0] += 1
        result = _real_sorted(iterable, *args, **kwargs)
        if call_count[0] == 1 and isinstance(iterable, set):
            return result + [999]  # inject foreign band to force mismatch
        return result

    return patched


# ---------------------------------------------------------------------------
# Mismatch tests — RED before Task 4 guard, GREEN after guard is added
# ---------------------------------------------------------------------------

def test_binary_mismatch_raises_value_error(tmp_path):
    """
    Binary path: patched sorted injects band 999 (not in original_bands=[10,20,30]).
    col_order will have 3 entries but selected_bands has 4 → guard must raise ValueError.
    """
    svc = _service()
    model = _binary_svc(n_features=3)
    out = tmp_path / "model.json"

    with mock.patch("services.learning_service.sorted", side_effect=_make_patched_sorted()):
        with pytest.raises(ValueError, match="weight column reorder mismatch"):
            svc.export_model(model, [10, 20, 30], str(out), total_bands=200,
                             preprocessing_config=[])


def test_multiclass_mismatch_raises_value_error(tmp_path):
    """
    Multiclass path: same patched-sorted mismatch scenario.
    """
    svc = _service()
    model = _multiclass_svc(n_features=3, n_classes=3)
    out = tmp_path / "model.json"

    with mock.patch("services.learning_service.sorted", side_effect=_make_patched_sorted()):
        with pytest.raises(ValueError, match="weight column reorder mismatch"):
            svc.export_model(model, [10, 20, 30], str(out), total_bands=200,
                             preprocessing_config=[])


# ---------------------------------------------------------------------------
# Happy-path test — always GREEN (no guard needed)
# ---------------------------------------------------------------------------

def test_binary_happy_path_no_exception(tmp_path):
    """
    Binary path happy path: original_bands=[30, 10, 20] (unsorted input),
    after sort+dedup selected_bands=[10, 20, 30]. All bands present in
    original_bands → col_order length == selected_bands length → no exception.
    Verify weight reorder is also correct.
    """
    svc = _service()
    # coef_ matches original band order: [30, 10, 20] → weights [1.0, 2.0, 3.0]
    model = LinearSVC()
    model.coef_ = np.array([[1.0, 2.0, 3.0]], dtype=float)  # for bands [30, 10, 20]
    model.intercept_ = np.array([0.0], dtype=float)
    model.classes_ = np.array([0, 1])

    out = tmp_path / "model.json"
    svc.export_model(model, [30, 10, 20], str(out), total_bands=200,
                     preprocessing_config=[])

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["SelectedBands"] == [10, 20, 30]
    # original=[30,10,20]: col 30→0 (w=1.0), col 10→1 (w=2.0), col 20→2 (w=3.0)
    # selected=[10,20,30]: reorder [1, 2, 0] → weights [2.0, 3.0, 1.0]
    assert data["Weights"] == pytest.approx([2.0, 3.0, 1.0])
    assert data["IsMultiClass"] is False
