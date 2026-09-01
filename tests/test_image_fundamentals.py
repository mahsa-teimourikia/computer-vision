import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


LESSON = Path("curriculum/beginner/01-image-fundamentals/lab.py")
SPEC = importlib.util.spec_from_file_location("image_fundamentals_lab", LESSON)
lab = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)


def test_baseline_meets_success_criterion():
    image, target = lab.make_scene()
    metrics = lab.evaluate(lab.segment(image, 0.50), target)
    assert metrics.intersection_over_union >= 0.95


def test_illumination_shift_exposes_and_mitigation_improves_failure():
    image, target = lab.make_scene()
    dark_image = np.clip(image * 0.65, 0.0, 1.0)
    shifted = lab.evaluate(lab.segment(dark_image, 0.50), target)
    mitigated = lab.evaluate(lab.segment(dark_image, 0.32), target)
    assert shifted.recall < 0.9
    assert mitigated.intersection_over_union > shifted.intersection_over_union


def test_segment_rejects_range_mismatch():
    image, _ = lab.make_scene()
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        lab.segment((image * 255).astype(np.uint8), 0.50)


def test_uint8_normalization_is_explicit():
    pixels = np.array([[0, 127, 255]], dtype=np.uint8)
    normalized = lab.normalize_uint8(pixels)
    np.testing.assert_allclose(normalized, [[0.0, 127 / 255, 1.0]])