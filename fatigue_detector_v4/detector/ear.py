"""Eye Aspect Ratio (EAR) calculation for drowsiness detection.

The EAR metric was introduced by Soukupová & Čech (2016) and provides a
real-time, per-frame measure of eye openness.  A sharp drop in EAR
indicates a blink or prolonged eye closure — a key fatigue indicator.

Typical values
--------------
* ~0.30 when eyes are wide open.
* ~0.05 when eyes are fully closed.

References
----------
Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using
Facial Landmarks. *21st Computer Vision Winter Workshop*.
"""

import numpy as np
from scipy.spatial import distance


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Calculate the Eye Aspect Ratio for a single eye.

    The function expects exactly **6** landmark points ordered as per
    the iBUG 300-W convention::

        P1(0)---P4(3)   (horizontal axis)
        P2(1)   P6(5)   (vertical pair 1)
        P3(2)   P5(4)   (vertical pair 2)

    .. math::

        EAR = \\frac{\\|P2 - P6\\| + \\|P3 - P5\\|}{2 \\cdot \\|P1 - P4\\|}

    Parameters
    ----------
    eye : np.ndarray
        Array of shape ``(6, 2)`` with the (x, y) coordinates of the
        six eye landmarks.

    Returns
    -------
    float
        Eye aspect ratio in the range [0, ~0.5].  Returns ``0.0`` when
        the horizontal distance is zero (degenerate input).
    """
    A: float = distance.euclidean(eye[1], eye[5])
    B: float = distance.euclidean(eye[2], eye[4])
    C: float = distance.euclidean(eye[0], eye[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def compute_ear(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """Compute the average EAR across both eyes.

    Averaging removes per-eye noise and minor facial asymmetry, giving
    a more robust drowsiness signal.

    Parameters
    ----------
    left_eye : np.ndarray
        Left eye landmarks, shape ``(6, 2)``.
    right_eye : np.ndarray
        Right eye landmarks, shape ``(6, 2)``.

    Returns
    -------
    float
        Mean eye aspect ratio of the two eyes.
    """
    left_ear: float = eye_aspect_ratio(left_eye)
    right_ear: float = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0
