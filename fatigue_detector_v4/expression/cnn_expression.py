"""Optional CNN-based expression detection using DeepFace.

Runs DeepFace emotion analysis on a background thread every N frames
to classify the driver's facial expression (happy, sad, angry, surprise,
fear, disgust, neutral) without blocking the main video loop.

This module degrades gracefully — if DeepFace is not installed, it simply
reports 'N/A' for the emotion and does nothing.
"""

import threading
import numpy as np


class ExpressionDetector:
    """Background CNN expression detector using DeepFace.

    Parameters
    ----------
    interval : int
        Analyze every Nth frame to reduce CPU/GPU load.
        Default is 5 (analyze 1 in 5 frames).
    """

    def __init__(self, interval: int = 5):
        self.interval = interval
        self.frame_count = 0
        self.current_emotion = "N/A"
        self.lock = threading.Lock()
        self._deepface = None
        self._available = False

        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._available = True
        except ImportError:
            print("[ExpressionDetector] DeepFace not installed. "
                  "Expression detection disabled.")

    @property
    def available(self) -> bool:
        """Whether DeepFace is installed and available."""
        return self._available

    def update(self, frame: np.ndarray) -> None:
        """Submit a frame for analysis (non-blocking).

        Only processes every Nth frame. Analysis runs in a
        background daemon thread.

        Parameters
        ----------
        frame : np.ndarray
            BGR video frame from OpenCV.
        """
        if not self._available:
            return

        self.frame_count += 1
        if self.frame_count % self.interval != 0:
            return

        # Copy frame for thread safety and run analysis in background
        frame_copy = frame.copy()
        t = threading.Thread(target=self._analyze, args=(frame_copy,), daemon=True)
        t.start()

    def _analyze(self, frame: np.ndarray) -> None:
        """Run DeepFace emotion analysis (called in background thread)."""
        try:
            result = self._deepface.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                emotion = result[0]['dominant_emotion']
            else:
                emotion = result['dominant_emotion']

            with self.lock:
                self.current_emotion = emotion

        except Exception:
            pass

    def get_emotion(self) -> str:
        """Get the most recently detected emotion (thread-safe).

        Returns
        -------
        str
            The dominant emotion label, or 'N/A' if not available.
        """
        with self.lock:
            return self.current_emotion
