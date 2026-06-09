"""Lightweight 1-D Kalman filter for smoothing noisy metric streams.

Provides a scalar Kalman filter (:class:`KalmanFilter1D`) and a
convenience wrapper (:class:`MetricSmoother`) that manages one filter
per named metric.  Designed for real-time smoothing of values derived
from an IP-camera RTSP stream on low-power hardware (Intel Pentium
N3700).

Typical usage
-------------
>>> smoother = MetricSmoother(['ear', 'mar'])
>>> smoothed = smoother.update(ear=0.31, mar=0.12)
>>> smoothed['ear']
0.31
"""

from __future__ import annotations

# Default metric names tracked in the driver fatigue pipeline.
DEFAULT_METRICS: list[str] = [
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "gaze_h",
    "gaze_v",
]


class KalmanFilter1D:
    """Scalar (1-D) Kalman filter for a single noisy signal.

    The filter maintains a running *estimate* and its associated
    *error covariance*.  Each call to :meth:`update` performs a
    predict-then-correct cycle and returns the smoothed value.

    Parameters
    ----------
    process_noise : float
        Variance added to the prediction error at every step (models
        how much the true value is expected to change between frames).
    measurement_noise : float
        Assumed variance of each incoming measurement.
    initial_estimate : float
        Starting value for the internal state.
    initial_error : float
        Starting value for the error covariance.
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 0.1,
        initial_estimate: float = 0.0,
        initial_error: float = 1.0,
    ) -> None:
        self._process_noise: float = process_noise
        self._measurement_noise: float = measurement_noise
        self._estimate: float = initial_estimate
        self._error: float = initial_error

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, measurement: float) -> float:
        """Run one predict → correct cycle and return the smoothed value.

        Parameters
        ----------
        measurement : float
            Raw (noisy) observation for the current frame.

        Returns
        -------
        float
            Filtered estimate after incorporating *measurement*.
        """
        # --- Predict step ---
        predicted_estimate: float = self._estimate
        predicted_error: float = self._error + self._process_noise

        # --- Update step ---
        kalman_gain: float = predicted_error / (
            predicted_error + self._measurement_noise
        )
        self._estimate = predicted_estimate + kalman_gain * (
            measurement - predicted_estimate
        )
        self._error = (1.0 - kalman_gain) * predicted_error
        return self._estimate

    def reset(self, value: float = 0.0) -> None:
        """Reset the filter state.

        Parameters
        ----------
        value : float
            New initial estimate (error covariance is reset to ``1.0``).
        """
        self._estimate = value
        self._error = 1.0

    @property
    def value(self) -> float:
        """Current smoothed estimate (read-only)."""
        return self._estimate


class MetricSmoother:
    """Convenience wrapper that manages one :class:`KalmanFilter1D` per metric.

    Parameters
    ----------
    metric_names : list[str] | None
        Names of the metrics to track.  Defaults to
        :data:`DEFAULT_METRICS` when ``None``.
    process_noise : float
        Passed through to each :class:`KalmanFilter1D`.
    measurement_noise : float
        Passed through to each :class:`KalmanFilter1D`.

    Examples
    --------
    >>> smoother = MetricSmoother(['ear', 'mar', 'gaze_h'])
    >>> result = smoother.update(ear=0.28, mar=0.10, gaze_h=0.52)
    >>> isinstance(result, dict)
    True
    """

    def __init__(
        self,
        metric_names: list[str] | None = None,
        process_noise: float = 1e-3,
        measurement_noise: float = 0.1,
    ) -> None:
        if metric_names is None:
            metric_names = list(DEFAULT_METRICS)

        self._filters: dict[str, KalmanFilter1D] = {
            name: KalmanFilter1D(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
            )
            for name in metric_names
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, **kwargs: float) -> dict[str, float]:
        """Feed new raw measurements and return all smoothed values.

        Only the metrics supplied as keyword arguments are updated; the
        rest retain their previous estimate.

        Parameters
        ----------
        **kwargs : float
            Metric name → raw measurement, e.g. ``ear=0.30``.

        Returns
        -------
        dict[str, float]
            Mapping of *every* tracked metric name to its current
            smoothed value.

        Raises
        ------
        KeyError
            If a keyword argument names a metric that was not
            registered during construction.
        """
        for name, measurement in kwargs.items():
            if name not in self._filters:
                raise KeyError(
                    f"Unknown metric '{name}'. "
                    f"Registered metrics: {list(self._filters)}"
                )
            self._filters[name].update(measurement)

        return {name: f.value for name, f in self._filters.items()}

    def get(self, name: str) -> float:
        """Return the current smoothed value for a single metric.

        Parameters
        ----------
        name : str
            Metric name (must have been registered at construction).

        Returns
        -------
        float
            Latest smoothed estimate.

        Raises
        ------
        KeyError
            If *name* is not a registered metric.
        """
        if name not in self._filters:
            raise KeyError(
                f"Unknown metric '{name}'. "
                f"Registered metrics: {list(self._filters)}"
            )
        return self._filters[name].value

    def reset(self) -> None:
        """Reset every filter to its initial state."""
        for f in self._filters.values():
            f.reset()
