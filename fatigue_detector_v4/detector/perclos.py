"""PERCLOS (Percentage of Eye Closure) calculation.

Implements a rolling window to track the fraction of time the driver's
eyes are closed. Industry standard metric: PERCLOS > 0.15 (eyes closed
15% of the time) indicates fatigue.
"""

from collections import deque


class PERCLOS:
    """Rolling window PERCLOS calculator.

    Tracks eye closure state over a fixed number of frames and computes
    the percentage of frames where the eyes were considered closed
    (EAR below threshold).

    Parameters
    ----------
    window_size : int
        Number of frames in the rolling window. Default is 60 frames
        (~2 seconds at 30 fps, or ~1 minute for slower analysis).

    Attributes
    ----------
    window : deque
        Circular buffer storing 1 (closed) or 0 (open) for each frame.

    Examples
    --------
    >>> perclos = PERCLOS(window_size=60)
    >>> value = perclos.update(ear=0.20, threshold=0.25)  # eyes closed
    >>> value = perclos.update(ear=0.30, threshold=0.25)  # eyes open
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)

    def update(self, ear: float, threshold: float) -> float:
        """Add a frame observation and return current PERCLOS value.

        Parameters
        ----------
        ear : float
            Current Eye Aspect Ratio value.
        threshold : float
            EAR threshold below which eyes are considered closed.

        Returns
        -------
        float
            Current PERCLOS value (0.0 to 1.0).
            PERCLOS > 0.15 indicates fatigue (industry standard).
        """
        self.window.append(1 if ear < threshold else 0)

        if len(self.window) == 0:
            return 0.0

        return sum(self.window) / len(self.window)

    def reset(self) -> None:
        """Clear the rolling window."""
        self.window.clear()

    @property
    def is_ready(self) -> bool:
        """Whether the window has been fully populated at least once."""
        return len(self.window) == self.window.maxlen

    @property
    def value(self) -> float:
        """Current PERCLOS value without updating."""
        if len(self.window) == 0:
            return 0.0
        return sum(self.window) / len(self.window)
