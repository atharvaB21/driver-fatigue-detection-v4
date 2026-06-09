"""Escalating alert system with visual overlays and threaded audio.

Production-level alert manager that provides:
- Visual feedback (colored overlays, banners, HUD) per scenario
- Audio alerts (beep sounds, alarm, text-to-speech) that escalate
- Distinct TTS messages for each of the 16 detection scenarios
- Grace period before audio fires to reduce false positives
- Rate-limited alerts to prevent spam and audio overlap
"""

import cv2
import numpy as np
import threading
import time
import os

# Optional audio dependencies — degrade gracefully if not installed
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import pygame
    #pygame.mixer.init()
    PYGAME_AVAILABLE = False
except (ImportError, Exception):
    PYGAME_AVAILABLE = False

from detector.decision import DriverState, DistractionType


# ──────────────────────── TTS Messages Per Scenario ────────────────────────

DISTRACTION_TTS_MESSAGES = {
    DistractionType.NONE: "",
    DistractionType.LOOKING_AWAY: "Eyes on the road!",
    DistractionType.REPEATED_GLANCES: "Please keep your eyes on the road.",
    DistractionType.PHONE_LOOKING: "Stop using your phone while driving!",
    DistractionType.TALKING_COPASSENGER: "Please focus on driving.",
    DistractionType.PHONE_CALL: "Please end the call and focus on driving.",
    DistractionType.NO_FACE: "Driver not detected. Please face the camera.",
    DistractionType.FACE_OBSCURED: "Camera view obstructed. Please clear the view.",
    DistractionType.SMOKING: "Smoking detected. Please keep both hands on the wheel.",
    DistractionType.EATING_DRINKING: "Please do not eat or drink while driving.",
    DistractionType.UNRESPONSIVE: "Driver unresponsive! Emergency alert!",
}

DISTRACTION_BANNER_TEXT = {
    DistractionType.NONE: "",
    DistractionType.LOOKING_AWAY: "EYES ON THE ROAD!",
    DistractionType.REPEATED_GLANCES: "KEEP EYES ON ROAD!",
    DistractionType.PHONE_LOOKING: "STOP USING PHONE!",
    DistractionType.TALKING_COPASSENGER: "FOCUS ON DRIVING!",
    DistractionType.PHONE_CALL: "END THE CALL!",
    DistractionType.NO_FACE: "NO FACE DETECTED",
    DistractionType.FACE_OBSCURED: "CAMERA OBSTRUCTED",
    DistractionType.SMOKING: "SMOKING DETECTED",
    DistractionType.EATING_DRINKING: "NO EATING/DRINKING!",
    DistractionType.UNRESPONSIVE: "!!! DRIVER UNRESPONSIVE !!!",
}

# Severity level per distraction type (affects color and urgency)
DISTRACTION_SEVERITY = {
    DistractionType.NONE: 0,
    DistractionType.TALKING_COPASSENGER: 1,   # Low — advisory
    DistractionType.SMOKING: 1,                # Low — advisory
    DistractionType.EATING_DRINKING: 1,        # Low — advisory
    DistractionType.REPEATED_GLANCES: 2,       # Medium — warning
    DistractionType.LOOKING_AWAY: 2,           # Medium — warning
    DistractionType.FACE_OBSCURED: 2,          # Medium — warning
    DistractionType.PHONE_CALL: 2,             # Medium — warning
    DistractionType.PHONE_LOOKING: 3,          # High — critical
    DistractionType.NO_FACE: 3,                # High — critical
    DistractionType.UNRESPONSIVE: 4,           # Emergency
}


class Alerter:
    """Escalating alert manager with visual and audio feedback.

    Visual alerts are drawn directly on OpenCV frames.
    Audio alerts run in daemon threads to avoid blocking the video loop.

    Parameters
    ----------
    sounds_dir : str
        Path to directory containing beep.wav and alarm.wav.
    tts_cooldown : float
        Minimum seconds between TTS alerts to prevent spam.
    grace_period : float
        Seconds of visual-only alert before audio fires.
    """

    # Color constants (BGR format for OpenCV)
    COLOR_GREEN   = (0, 200, 0)
    COLOR_AMBER   = (0, 191, 255)
    COLOR_RED     = (0, 0, 255)
    COLOR_ORANGE  = (0, 140, 255)
    COLOR_YELLOW  = (0, 255, 255)
    COLOR_WHITE   = (255, 255, 255)
    COLOR_BLACK   = (0, 0, 0)
    COLOR_DARK_BG = (40, 40, 40)
    COLOR_PURPLE  = (180, 0, 180)

    def __init__(self, sounds_dir: str, tts_cooldown: float = 5.0,
                 grace_period: float = 1.0):
        self.sounds_dir = sounds_dir
        self.tts_cooldown = tts_cooldown
        self.grace_period = grace_period
        self.last_tts_time = 0.0
        self.last_beep_time = 0.0

        # Track when each alert first appeared (for grace period)
        self._alert_start_times = {}

        # Initialize TTS engine
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 1.0)
            except Exception:
                self.tts_engine = None

    # ──────────────────────── Visual Overlays ────────────────────────

    def draw_overlay(self, frame: np.ndarray, state: DriverState,
                     distraction_type: DistractionType, yawning: bool,
                     ear: float, mar: float, perclos: float,
                     pitch: float, yaw: float, fps: float,
                     gaze_h: float = 0.5, gaze_v: float = 0.5) -> np.ndarray:
        """Draw all visual alerts and HUD on the video frame.

        Parameters
        ----------
        frame : np.ndarray
            The video frame to draw on (modified in-place and returned).
        state : DriverState
            Current driver state from the decision engine.
        distraction_type : DistractionType
            Current distraction type detected.
        yawning : bool
            Whether the driver is currently yawning.
        ear, mar, perclos, pitch, yaw, fps : float
            Current metric values for HUD display.
        gaze_h, gaze_v : float
            Horizontal and vertical gaze ratios for HUD.

        Returns
        -------
        np.ndarray
            The frame with overlays drawn.
        """
        h, w = frame.shape[:2]

        # ── State-specific overlays (drowsiness) ──
        if state == DriverState.ASLEEP:
            # Flashing red border
            flash = int(time.time() * 4) % 2 == 0
            border_color = self.COLOR_RED if flash else self.COLOR_BLACK
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 8)
            self._draw_alert_banner(frame, "!!! WAKE UP !!!", self.COLOR_RED, large=True)

        elif state == DriverState.VERY_DROWSY:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), self.COLOR_RED, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), self.COLOR_RED, 4)
            self._draw_alert_banner(frame, "VERY DROWSY - PULL OVER!", self.COLOR_RED)

        elif state == DriverState.DROWSY:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), self.COLOR_AMBER, -1)
            cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
            self._draw_alert_banner(frame, "DROWSY - STAY ALERT", self.COLOR_AMBER)

        else:
            self._draw_status_badge(frame, "ALERT", self.COLOR_GREEN)

        # ── Distraction overlay ──
        if distraction_type != DistractionType.NONE:
            severity = DISTRACTION_SEVERITY.get(distraction_type, 1)
            banner_text = DISTRACTION_BANNER_TEXT.get(
                distraction_type, "DISTRACTED"
            )

            if distraction_type == DistractionType.UNRESPONSIVE:
                # Emergency — flashing red like ASLEEP
                flash = int(time.time() * 4) % 2 == 0
                border_color = self.COLOR_RED if flash else self.COLOR_BLACK
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 8)
                self._draw_alert_banner(
                    frame, banner_text, self.COLOR_RED,
                    y_offset=80, large=True
                )
            elif severity >= 3:
                # High severity — red banner with beep
                self._draw_alert_banner(
                    frame, banner_text, self.COLOR_RED, y_offset=80
                )
            elif severity >= 2:
                # Medium — orange banner
                self._draw_alert_banner(
                    frame, banner_text, self.COLOR_ORANGE, y_offset=80
                )
            else:
                # Low — yellow banner
                self._draw_alert_banner(
                    frame, banner_text, self.COLOR_YELLOW, y_offset=80
                )

        # ── Yawning label ──
        if yawning:
            self._draw_alert_banner(
                frame, "YAWNING DETECTED", self.COLOR_YELLOW, y_offset=120
            )

        # ── HUD Panel ──
        self._draw_hud(
            frame, ear, mar, perclos, pitch, yaw, fps, state,
            gaze_h, gaze_v, distraction_type
        )

        return frame

    def _draw_alert_banner(self, frame: np.ndarray, text: str, color: tuple,
                           y_offset: int = 30, large: bool = False) -> None:
        """Draw a centered alert banner with background."""
        h, w = frame.shape[:2]
        font_scale = 1.2 if large else 0.8
        thickness = 3 if large else 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = y_offset + text_size[1]
        pad = 10

        # Dark background with colored border
        cv2.rectangle(frame,
                      (x - pad, y - text_size[1] - pad),
                      (x + text_size[0] + pad, y + pad),
                      self.COLOR_DARK_BG, -1)
        cv2.rectangle(frame,
                      (x - pad, y - text_size[1] - pad),
                      (x + text_size[0] + pad, y + pad),
                      color, 2)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def _draw_status_badge(self, frame: np.ndarray, text: str, color: tuple) -> None:
        """Draw a small status badge in the top-left corner."""
        cv2.rectangle(frame, (10, 10), (130, 45), self.COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (10, 10), (130, 45), color, 2)
        cv2.putText(frame, text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_hud(self, frame: np.ndarray, ear: float, mar: float,
                  perclos: float, pitch: float, yaw: float,
                  fps: float, state: DriverState,
                  gaze_h: float = 0.5, gaze_v: float = 0.5,
                  distraction_type: DistractionType = DistractionType.NONE
                  ) -> None:
        """Draw a semi-transparent HUD panel with real-time metrics."""
        h, w = frame.shape[:2]
        panel_w, panel_h = 230, 260
        x0 = w - panel_w - 10
        y0 = 10

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      self.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (100, 100, 100), 1)

        # Metric rows with color-coded values
        metrics = [
            (f"FPS: {fps:.0f}",         self.COLOR_WHITE),
            (f"EAR: {ear:.3f}",          self.COLOR_GREEN if ear > 0.25 else self.COLOR_RED),
            (f"MAR: {mar:.3f}",          self.COLOR_GREEN if mar < 0.60 else self.COLOR_YELLOW),
            (f"PERCLOS: {perclos:.2f}",  self.COLOR_GREEN if perclos < 0.15 else self.COLOR_RED),
            (f"Pitch: {pitch:.1f}",      self.COLOR_GREEN if abs(pitch) < 20 else self.COLOR_ORANGE),
            (f"Yaw: {yaw:.1f}",          self.COLOR_GREEN if abs(yaw) < 30 else self.COLOR_ORANGE),
            (f"Gaze H: {gaze_h:.2f}",   self.COLOR_GREEN if 0.3 < gaze_h < 0.7 else self.COLOR_ORANGE),
            (f"Gaze V: {gaze_v:.2f}",    self.COLOR_GREEN if gaze_v < 0.65 else self.COLOR_ORANGE),
            (f"State: {state.name}",     self._state_color(state)),
        ]

        # Add distraction type if active
        if distraction_type != DistractionType.NONE:
            severity = DISTRACTION_SEVERITY.get(distraction_type, 1)
            dist_color = (self.COLOR_RED if severity >= 3
                          else self.COLOR_ORANGE if severity >= 2
                          else self.COLOR_YELLOW)
            metrics.append(
                (f"Dist: {distraction_type.name}", dist_color)
            )

        for i, (text, color) in enumerate(metrics):
            cv2.putText(frame, text, (x0 + 10, y0 + 25 + i * 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _state_color(self, state: DriverState) -> tuple:
        """Get the display color for a given driver state."""
        return {
            DriverState.ALERT:       self.COLOR_GREEN,
            DriverState.DROWSY:      self.COLOR_AMBER,
            DriverState.VERY_DROWSY: self.COLOR_RED,
            DriverState.ASLEEP:      self.COLOR_RED,
        }.get(state, self.COLOR_WHITE)

    # ──────────────────────── Audio Alerts ────────────────────────

    def play_alert(self, state: DriverState,
                   distraction_type: DistractionType = DistractionType.NONE
                   ) -> None:
        """Play appropriate audio alert based on state and distraction.

        Implements a grace period: visual-only for the first N seconds,
        then escalates to audio. Audio is rate-limited to prevent overlap.

        Parameters
        ----------
        state : DriverState
            Current driver state.
        distraction_type : DistractionType
            Current distraction type.
        """
        now = time.time()

        # ── Drowsiness alerts ──
        if state == DriverState.ASLEEP:
            if self._past_grace("asleep", now):
                if now - self.last_tts_time > self.tts_cooldown:
                    self._speak_async("Wake up! Pull over immediately!")
                    self.last_tts_time = now
                self._play_sound_async(
                    os.path.join(self.sounds_dir, 'alarm.wav')
                )
        elif state == DriverState.VERY_DROWSY:
            if self._past_grace("very_drowsy", now):
                if now - self.last_tts_time > self.tts_cooldown:
                    self._speak_async(
                        "Warning! You are very drowsy. Pull over when safe."
                    )
                    self.last_tts_time = now
                if now - self.last_beep_time > 2.0:
                    self._play_sound_async(
                        os.path.join(self.sounds_dir, 'beep.wav')
                    )
                    self.last_beep_time = now
        elif state == DriverState.DROWSY:
            if self._past_grace("drowsy", now):
                if now - self.last_tts_time > self.tts_cooldown:
                    self._speak_async(
                        "You seem drowsy. Please stay alert."
                    )
                    self.last_tts_time = now
        else:
            # Clear drowsiness grace timers
            for key in ["asleep", "very_drowsy", "drowsy"]:
                self._alert_start_times.pop(key, None)

        # ── Distraction alerts ──
        if distraction_type != DistractionType.NONE:
            grace_key = f"dist_{distraction_type.name}"
            severity = DISTRACTION_SEVERITY.get(distraction_type, 1)

            if self._past_grace(grace_key, now):
                # TTS message
                tts_msg = DISTRACTION_TTS_MESSAGES.get(distraction_type, "")
                if tts_msg and now - self.last_tts_time > self.tts_cooldown:
                    self._speak_async(tts_msg)
                    self.last_tts_time = now

                # Beep for medium+ severity
                if severity >= 2 and now - self.last_beep_time > 2.0:
                    self._play_sound_async(
                        os.path.join(self.sounds_dir, 'beep.wav')
                    )
                    self.last_beep_time = now

                # Alarm for emergency (unresponsive)
                if severity >= 4:
                    self._play_sound_async(
                        os.path.join(self.sounds_dir, 'alarm.wav')
                    )
        else:
            # Clear distraction grace timers
            keys_to_clear = [
                k for k in self._alert_start_times if k.startswith("dist_")
            ]
            for k in keys_to_clear:
                del self._alert_start_times[k]

    def play_no_face_alert(self) -> None:
        """Play audio alert for no face detected (convenience wrapper)."""
        self.play_alert(DriverState.ALERT, DistractionType.NO_FACE)

    def _past_grace(self, alert_key: str, now: float) -> bool:
        """Check if an alert has been active past the grace period.

        Returns True if the alert has been active longer than
        ``self.grace_period`` seconds, meaning audio should fire.
        """
        if alert_key not in self._alert_start_times:
            self._alert_start_times[alert_key] = now
            return False
        return (now - self._alert_start_times[alert_key]) >= self.grace_period

    def _play_sound_async(self, path: str) -> None:
        """Play a WAV file asynchronously using pygame."""
        if not PYGAME_AVAILABLE or not os.path.exists(path):
            return

        def _play():
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _speak_async(self, text: str) -> None:
        """Speak text asynchronously using pyttsx3 TTS."""
        if self.tts_engine is None:
            return

        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception:
                pass

        threading.Thread(target=_speak, daemon=True).start()

    # ──────────────────────── Cleanup ────────────────────────

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
