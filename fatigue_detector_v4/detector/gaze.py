"""Iris-based gaze direction estimation using MediaPipe Face Mesh.

Leverages the 478-landmark Face Mesh model which includes 10 iris
landmarks (5 per eye) at zero additional CPU cost.  Horizontal and
vertical gaze ratios are derived from the position of each iris centre
relative to the surrounding eye corners / eyelid landmarks.

Gaze ratios
-----------
* ``gaze_h``: 0.0 = far left, 0.5 = centre, 1.0 = far right.
* ``gaze_v``: 0.0 = looking up, 0.5 = centre, 1.0 = looking down.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Iris landmark indices (MediaPipe Face Mesh 478-point model)
# ---------------------------------------------------------------------------
LEFT_IRIS_CENTER: int = 468
RIGHT_IRIS_CENTER: int = 473

# ---------------------------------------------------------------------------
# Eye corner indices for horizontal gaze ratio
# ---------------------------------------------------------------------------
LEFT_EYE_INNER: int = 33
LEFT_EYE_OUTER: int = 133
RIGHT_EYE_INNER: int = 362
RIGHT_EYE_OUTER: int = 263

# ---------------------------------------------------------------------------
# Eye vertical indices for vertical gaze ratio
# ---------------------------------------------------------------------------
LEFT_EYE_TOP: int = 159
LEFT_EYE_BOTTOM: int = 145
RIGHT_EYE_TOP: int = 386
RIGHT_EYE_BOTTOM: int = 374


class GazeEstimator:
    """Estimate horizontal and vertical gaze direction from iris landmarks.

    The estimator is stateless — each call to :meth:`estimate` is
    independent.  This keeps the class lightweight and suitable for
    low-power hardware (Intel Pentium N3700).

    Examples
    --------
    >>> import numpy as np
    >>> estimator = GazeEstimator()
    >>> landmarks = np.zeros((478, 2), dtype=np.int32)
    >>> gaze_h, gaze_v = estimator.estimate(landmarks)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, landmarks: np.ndarray) -> tuple[float, float]:
        """Compute horizontal and vertical gaze ratios.

        Parameters
        ----------
        landmarks : np.ndarray
            Full 478-point landmark array of shape ``(478, 2)`` with
            integer pixel coordinates (dtype ``int32``).

        Returns
        -------
        tuple[float, float]
            ``(gaze_h, gaze_v)`` each in the range [0.0, 1.0].
            Returns ``(0.5, 0.5)`` when the landmark array has fewer
            than 478 points (no iris data available).
        """
        if landmarks.shape[0] < 478:
            return (0.5, 0.5)

        gaze_h: float = self._horizontal_ratio(landmarks)
        gaze_v: float = self._vertical_ratio(landmarks)
        return (gaze_h, gaze_v)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _horizontal_ratio(landmarks: np.ndarray) -> float:
        """Average left/right horizontal iris position.

        For each eye the ratio is::

            (iris_center_x - inner_corner_x) / (outer_corner_x - inner_corner_x)

        A value of 0.0 means the iris sits at the inner corner (far
        left for the left eye) and 1.0 at the outer corner.
        """
        left_ratio: float = _safe_ratio(
            landmarks[LEFT_IRIS_CENTER][0],
            landmarks[LEFT_EYE_INNER][0],
            landmarks[LEFT_EYE_OUTER][0],
        )
        right_ratio: float = _safe_ratio(
            landmarks[RIGHT_IRIS_CENTER][0],
            landmarks[RIGHT_EYE_INNER][0],
            landmarks[RIGHT_EYE_OUTER][0],
        )
        return (left_ratio + right_ratio) / 2.0

    @staticmethod
    def _vertical_ratio(landmarks: np.ndarray) -> float:
        """Average left/right vertical iris position.

        For each eye the ratio is::

            (iris_center_y - top_y) / (bottom_y - top_y)

        A value of 0.0 means the iris sits at the top eyelid (looking
        up) and 1.0 at the bottom eyelid (looking down).
        """
        left_ratio: float = _safe_ratio(
            landmarks[LEFT_IRIS_CENTER][1],
            landmarks[LEFT_EYE_TOP][1],
            landmarks[LEFT_EYE_BOTTOM][1],
        )
        right_ratio: float = _safe_ratio(
            landmarks[RIGHT_IRIS_CENTER][1],
            landmarks[RIGHT_EYE_TOP][1],
            landmarks[RIGHT_EYE_BOTTOM][1],
        )
        return (left_ratio + right_ratio) / 2.0


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _safe_ratio(value: int, ref_start: int, ref_end: int) -> float:
    """Compute ``(value - ref_start) / (ref_end - ref_start)`` safely.

    Returns ``0.5`` (centre) when the denominator is zero to avoid
    division errors on degenerate inputs.

    Parameters
    ----------
    value : int
        Numerator reference (e.g. iris centre coordinate).
    ref_start : int
        Start of the normalisation range (e.g. inner corner / top lid).
    ref_end : int
        End of the normalisation range (e.g. outer corner / bottom lid).

    Returns
    -------
    float
        Normalised ratio, or ``0.5`` on zero-width range.
    """
    denom: int = ref_end - ref_start
    if denom == 0:
        return 0.5
    return float(value - ref_start) / float(denom)
