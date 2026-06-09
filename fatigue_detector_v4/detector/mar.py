"""Mouth Aspect Ratio (MAR) calculation for yawn detection.

MAR is the oral analogue of EAR: it measures how wide the mouth is
opened relative to its width.  A sustained high MAR value indicates a
yawn, which is a strong predictor of driver fatigue.

Typical values
--------------
* ~0.1-0.2  normal / talking.
* ≥0.6      yawning.
"""

import numpy as np


def _fast_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Fast Euclidean distance without scipy overhead."""
    d = p1 - p2
    return float(np.sqrt(d[0] * d[0] + d[1] * d[1]))


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """Calculate the Mouth Aspect Ratio from inner-mouth landmarks.

    The function expects the **8 inner-mouth** landmarks (iBUG indices
    60-67) in their canonical order.  Three vertical distances are
    averaged and normalised by the horizontal mouth width::

        MAR = (|61-67| + |62-66| + |63-65|) / (3 · |60-64|)

    Using 0-indexed positions within the 8-point array:

    * Vertical pairs: (1, 7), (2, 6), (3, 5)
    * Horizontal pair: (0, 4)

    Parameters
    ----------
    mouth : np.ndarray
        Inner-mouth landmarks of shape ``(8, 2)``, corresponding to
        iBUG points 60-67.

    Returns
    -------
    float
        Mouth aspect ratio.  Returns ``0.0`` when the horizontal
        distance is zero (degenerate input).
    """
    A: float = _fast_dist(mouth[1], mouth[7])   # 61, 67
    B: float = _fast_dist(mouth[2], mouth[6])   # 62, 66
    C: float = _fast_dist(mouth[3], mouth[5])   # 63, 65
    D: float = _fast_dist(mouth[0], mouth[4])   # 60, 64

    if D == 0:
        return 0.0

    return (A + B + C) / (3.0 * D)
