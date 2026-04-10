"""
Doc-anchor tests for the total_bands required-invariant in export_model().

These tests encode the documented contract:
    "total_bands is required. Passing None (or a non-positive value) raises
    ValueError. The error message must contain 'total_bands' to be meaningful."

Semantic angle covered here (NOT duplicated from test_export_required_raw_bands.py):
  - Error message content: confirms the ValueError message references 'total_bands'
  - Edge cases: total_bands=0 and total_bands=-1 (non-positive) also raise
  - Happy path: total_bands=200 does not raise

Implementation note (learning_service.py lines 516-522):
    if total_bands is not None and total_bands > 0:
        ...clamp...
    else:
        raise ValueError("export_model: total_bands는 필수입니다. ...")
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


def call_export(total_bands, tmp_path):
    """Helper: minimal export_model() call with a 2-feature binary SVC."""
    svc = _service()
    model = LinearSVC()
    model.coef_ = np.ones((1, 2), dtype=float)
    model.intercept_ = np.array([0.0], dtype=float)
    model.classes_ = np.array([0, 1])

    out = str(tmp_path / "model.json")
    svc.export_model(
        model, [10, 20], out,
        preprocessing_config=[],
        total_bands=total_bands,
    )
    return json.load(open(out, encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests: invalid total_bands values
# ---------------------------------------------------------------------------

def test_total_bands_none_error_message_contains_key(tmp_path):
    """
    total_bands=None raises ValueError whose message contains 'total_bands'.
    The message must be meaningful, not a bare assert or unrelated text.
    """
    with pytest.raises(ValueError) as exc_info:
        call_export(total_bands=None, tmp_path=tmp_path)
    assert "total_bands" in str(exc_info.value)


def test_total_bands_zero_raises(tmp_path):
    """
    total_bands=0 must raise ValueError.
    The condition 'total_bands is not None and total_bands > 0' is False for 0.
    """
    with pytest.raises(ValueError):
        call_export(total_bands=0, tmp_path=tmp_path)


def test_total_bands_negative_raises(tmp_path):
    """
    total_bands=-1 must raise ValueError.
    Negative band counts are physically invalid.
    """
    with pytest.raises(ValueError):
        call_export(total_bands=-1, tmp_path=tmp_path)


# ---------------------------------------------------------------------------
# Happy path: valid total_bands
# ---------------------------------------------------------------------------

def test_total_bands_valid_does_not_raise(tmp_path):
    """total_bands=200 (valid positive integer) must not raise."""
    data = call_export(total_bands=200, tmp_path=tmp_path)
    # Sanity check: RequiredRawBands is present and non-empty
    assert "RequiredRawBands" in data
    assert len(data["RequiredRawBands"]) > 0
