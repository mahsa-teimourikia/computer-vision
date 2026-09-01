"""Deterministic image-fundamentals lab utilities.

The functions deliberately use only NumPy so learners can inspect every
transformation without a vision framework hiding array, dtype, or range rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class SegmentationMetrics:
    """Pixel-level metrics for a binary foreground mask."""

    intersection_over_union: float
    precision: float
    recall: float


def make_scene(size: int = 64, noise: float = 0.04, seed: int = 7) -> tuple[FloatImage, Mask]:
    """Create a grayscale scene containing a bright circular foreground object."""

    if size < 16:
        raise ValueError("size must be at least 16 pixels")
    if not 0.0 <= noise <= 0.25:
        raise ValueError("noise must be between 0 and 0.25")

    rows, columns = np.ogrid[:size, :size]
    radius = size * 0.22
    target = (rows - size / 2) ** 2 + (columns - size / 2) ** 2 <= radius**2

    image = np.full((size, size), 0.2, dtype=np.float64)
    image[target] = 0.78
    rng = np.random.default_rng(seed)
    image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 1.0), target


def normalize_uint8(image: NDArray[np.uint8]) -> FloatImage:
    """Convert a uint8 image from [0, 255] to float64 in [0, 1]."""

    if image.dtype != np.uint8:
        raise TypeError("normalize_uint8 expects an array with dtype uint8")
    return image.astype(np.float64) / 255.0


def segment(image: FloatImage, threshold: float) -> Mask:
    """Return pixels at or above a validated intensity threshold."""

    if image.ndim != 2:
        raise ValueError("segment expects a two-dimensional grayscale image")
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("image values must be finite and in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return image >= threshold


def evaluate(prediction: Mask, target: Mask) -> SegmentationMetrics:
    """Compute IoU, precision, and recall with safe zero-denominator rules."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = np.logical_and(prediction, target).sum()
    predicted_positive = prediction.sum()
    actual_positive = target.sum()
    union = np.logical_or(prediction, target).sum()

    return SegmentationMetrics(
        intersection_over_union=float(true_positive / union) if union else 1.0,
        precision=float(true_positive / predicted_positive) if predicted_positive else 0.0,
        recall=float(true_positive / actual_positive) if actual_positive else 0.0,
    )


def sweep_thresholds(
    image: FloatImage, target: Mask, thresholds: list[float]
) -> list[tuple[float, SegmentationMetrics]]:
    """Evaluate multiple thresholds while preserving their input order."""

    return [(threshold, evaluate(segment(image, threshold), target)) for threshold in thresholds]
