"""Decision engine with Finite State Machine for driver state classification.

Production-level 16-scenario FSM with:
- 4-level drowsiness states (ALERT → DROWSY → VERY_DROWSY → ASLEEP)
- 9 distraction types (looking away, VATS, phone, co-passenger, etc.)
- Yawning detection
- Smoking / eating heuristics
- Unresponsive driver detection

State transitions are based on EAR, MAR, PERCLOS, head pose angles,
and iris-based gaze direction. All timing thresholds are configurable
via config.py.
"""

from enum import IntEnum
import time
from collections import deque


class DriverState(IntEnum):
    """Enumeration of possible driver alertness states.

    States are ordered by severity so comparisons like
    ``state >= DriverState.DROWSY`` work intuitively.
    """
    ALERT = 0
    DROWSY = 1
    VERY_DROWSY = 2
    ASLEEP = 3


class DistractionType(IntEnum):
    """Enumeration of distraction / risky behaviour types.

    Used to provide scenario-specific alerts and logging.
    """
    NONE = 0
    LOOKING_AWAY = 1          # Scenario 7: sustained yaw > 30° for 3s+
    REPEATED_GLANCES = 2      # Scenario 8: >=3 short glances in 10s (VATS)
    PHONE_LOOKING = 3         # Scenario 9: gaze down + pitch < -15°
    TALKING_COPASSENGER = 4   # Scenario 10: yaw toward passenger 3s+
    PHONE_CALL = 5            # Scenario 11: yaw 15-35° + gaze off-center
    NO_FACE = 6               # Scenario 12: no face detected 2s+
    FACE_OBSCURED = 7         # Scenario 13: low confidence 3s+
    SMOKING = 8               # Scenario 14: MAR oscillation pattern
    EATING_DRINKING = 9       # Scenario 15: head back + wide mouth
    UNRESPONSIVE = 10         # Scenario 16: all metrics static 15s+


class DecisionEngine:
    """Finite State Machine for driver fatigue and distraction detection.

    Production-level engine supporting 16 alert scenarios with
    configurable thresholds and timing parameters.

    Parameters
    ----------
    ear_threshold : float
        EAR value below which eyes are considered closed.
    drowsy_frames : int
        Consecutive frames below EAR threshold to trigger DROWSY.
    very_drowsy_frames : int
        Consecutive frames below EAR threshold to trigger VERY_DROWSY.
    asleep_frames : int
        Consecutive frames below EAR threshold to trigger ASLEEP.
    mar_threshold : float
        MAR value above which the mouth is considered yawning.
    mar_consec_frames : int
        Consecutive frames above MAR threshold to confirm a yawn.
    pitch_threshold : float
        Head pitch angle (degrees) beyond which driver is distracted.
    yaw_threshold : float
        Head yaw angle (degrees) beyond which driver is looking away.
    perclos_threshold : float
        PERCLOS value above which fatigue is indicated.
    yaw_glance_threshold : float
        Yaw (degrees) for counting short glances (VATS).
    yaw_copassenger_threshold : float
        Yaw (degrees) toward passenger side for co-passenger detection.
    distraction_long_secs : float
        Seconds of sustained looking-away for long distraction.
    vats_count : int
        Number of short glances in window to trigger VATS alert.
    vats_window : float
        Window size (seconds) for VATS glance counting.
    phone_look_secs : float
        Seconds of looking down to trigger phone-looking alert.
    copassenger_secs : float
        Seconds of head turned to passenger to trigger co-passenger alert.
    phone_call_secs : float
        Seconds of head tilt + off-center gaze for phone-call alert.
    unresponsive_secs : float
        Seconds of static metrics to trigger unresponsive alert.
    smoking_secs : float
        Seconds of MAR oscillation to trigger smoking alert.
    eating_secs : float
        Seconds of head-back + mouth open for eating/drinking alert.
    face_obscured_secs : float
        Seconds of low face confidence for face-obscured alert.
    no_face_secs : float
        Seconds of no face detected for no-face alert.
    copassenger_side : str
        Which side the co-passenger is on ('right' or 'left').
    """

    def __init__(
        self,
        ear_threshold: float = 0.25,
        drowsy_frames: int = 20,
        very_drowsy_frames: int = 50,
        asleep_frames: int = 100,
        mar_threshold: float = 0.45,
        mar_consec_frames: int = 6,
        pitch_threshold: float = 20.0,
        yaw_threshold: float = 30.0,
        perclos_threshold: float = 0.15,
        yaw_glance_threshold: float = 20.0,
        yaw_copassenger_threshold: float = 25.0,
        distraction_long_secs: float = 3.0,
        vats_count: int = 3,
        vats_window: float = 10.0,
        phone_look_secs: float = 2.0,
        copassenger_secs: float = 3.0,
        phone_call_secs: float = 5.0,
        unresponsive_secs: float = 15.0,
        smoking_secs: float = 5.0,
        eating_secs: float = 2.0,
        face_obscured_secs: float = 3.0,
        no_face_secs: float = 2.0,
        copassenger_side: str = "right",
    ):
        # ── EAR / drowsiness thresholds ──
        self.ear_threshold = ear_threshold
        self.drowsy_frames = drowsy_frames
        self.very_drowsy_frames = very_drowsy_frames
        self.asleep_frames = asleep_frames

        # ── MAR / yawn thresholds ──
        self.mar_threshold = mar_threshold
        self.mar_consec_frames = mar_consec_frames

        # ── Head pose thresholds ──
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.yaw_glance_threshold = yaw_glance_threshold
        self.yaw_copassenger_threshold = yaw_copassenger_threshold

        # ── PERCLOS ──
        self.perclos_threshold = perclos_threshold

        # ── Distraction timing ──
        self.distraction_long_secs = distraction_long_secs
        self.vats_count = vats_count
        self.vats_window = vats_window
        self.phone_look_secs = phone_look_secs
        self.copassenger_secs = copassenger_secs
        self.phone_call_secs = phone_call_secs
        self.unresponsive_secs = unresponsive_secs
        self.smoking_secs = smoking_secs
        self.eating_secs = eating_secs
        self.face_obscured_secs = face_obscured_secs
        self.no_face_secs = no_face_secs

        # ── Co-passenger direction ──
        # For right-side passenger (left-hand drive): positive yaw
        # For left-side passenger (right-hand drive): negative yaw
        self.copassenger_yaw_sign = 1.0 if copassenger_side == "right" else -1.0

        # ── Initialize state ──
        self.reset()

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.state = DriverState.ALERT
        self.distraction_type = DistractionType.NONE
        self.ear_counter = 0
        self.yawn_counter = 0
        self.yawning = False
        self.last_state_change = time.time()

        # ── Distraction timers ──
        self._looking_away_start = None
        self._copassenger_start = None
        self._phone_look_start = None
        self._phone_call_start = None
        self._face_obscured_start = None
        self._no_face_start = None
        self._unresponsive_start = None
        self._smoking_start = None
        self._eating_start = None

        # ── VATS (Visual Attention Time Sharing) tracking ──
        self._glance_timestamps = deque()  # timestamps of short glances

        # ── Smoking detection: MAR history for oscillation detection ──
        self._mar_history = deque(maxlen=150)  # ~3-5 seconds of MAR values
        self._mar_time_history = deque(maxlen=150)

        # ── Unresponsive detection: metric variance tracking ──
        self._metric_history = deque(maxlen=30)  # recent metric snapshots

    def update(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        perclos: float,
        gaze_h: float = 0.5,
        gaze_v: float = 0.5,
        face_detected: bool = True,
        face_confidence: float = 1.0,
    ) -> tuple:
        """Update the state machine with new sensor readings.

        Parameters
        ----------
        ear : float
            Current average Eye Aspect Ratio.
        mar : float
            Current Mouth Aspect Ratio.
        pitch : float
            Current head pitch angle in degrees.
        yaw : float
            Current head yaw angle in degrees.
        perclos : float
            Current PERCLOS value (0.0 to 1.0).
        gaze_h : float
            Horizontal gaze ratio (0.0=left, 0.5=center, 1.0=right).
        gaze_v : float
            Vertical gaze ratio (0.0=up, 0.5=center, 1.0=down).
        face_detected : bool
            Whether a face was detected in the current frame.
        face_confidence : float
            Face detection confidence (0.0 to 1.0).

        Returns
        -------
        tuple
            (state: DriverState, distraction_type: DistractionType,
             is_yawning: bool)
        """
        now = time.time()

        # ── Handle no-face / face-obscured first ──
        if not face_detected:
            distraction = self._check_no_face(now)
            if distraction != DistractionType.NONE:
                self.distraction_type = distraction
                return self.state, self.distraction_type, self.yawning
            # No face but under threshold — keep previous state
            self.distraction_type = DistractionType.NONE
            return self.state, self.distraction_type, self.yawning
        else:
            self._no_face_start = None

        # ── Face obscured (low confidence) ──
        if face_confidence < 0.5:
            distraction = self._check_face_obscured(now)
            if distraction != DistractionType.NONE:
                self.distraction_type = distraction
                return self.state, self.distraction_type, self.yawning
        else:
            self._face_obscured_start = None

        # ══════════════════════════════════════════════
        #  EAR-based drowsiness state transitions
        # ══════════════════════════════════════════════
        if ear < self.ear_threshold:
            self.ear_counter += 1
        else:
            self.ear_counter = 0
            if self.state != DriverState.ALERT:
                self.state = DriverState.ALERT
                self.last_state_change = now

        # Escalate state based on consecutive closed-eye frames
        if self.ear_counter >= self.asleep_frames:
            if self.state != DriverState.ASLEEP:
                self.state = DriverState.ASLEEP
                self.last_state_change = now
        elif self.ear_counter >= self.very_drowsy_frames:
            if self.state != DriverState.VERY_DROWSY:
                self.state = DriverState.VERY_DROWSY
                self.last_state_change = now
        elif self.ear_counter >= self.drowsy_frames:
            if self.state != DriverState.DROWSY:
                self.state = DriverState.DROWSY
                self.last_state_change = now

        # ── PERCLOS can also trigger drowsiness ──
        if perclos > self.perclos_threshold and self.state == DriverState.ALERT:
            self.state = DriverState.DROWSY
            self.last_state_change = now

        # ══════════════════════════════════════════════
        #  MAR-based yawn detection
        # ══════════════════════════════════════════════
        if mar > self.mar_threshold:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.mar_consec_frames:
                self.yawning = True   # Confirmed yawn
            else:
                self.yawning = False
            self.yawn_counter = 0

        # Also flag yawning while mouth is still open
        if self.yawn_counter >= self.mar_consec_frames:
            self.yawning = True

        # ══════════════════════════════════════════════
        #  Distraction detection (priority-ordered)
        # ══════════════════════════════════════════════

        # Track MAR history for smoking detection
        self._mar_history.append(mar)
        self._mar_time_history.append(now)

        # Track metric history for unresponsive detection
        self._metric_history.append((ear, mar, pitch, yaw, gaze_h, gaze_v))

        # Reset distraction type — will be set below if any distraction found
        detected_distraction = DistractionType.NONE

        # ── Scenario 9: Phone looking (gaze down + head pitched down) ──
        if gaze_v > 0.65 and pitch < -15.0:
            if self._phone_look_start is None:
                self._phone_look_start = now
            elif now - self._phone_look_start >= self.phone_look_secs:
                detected_distraction = DistractionType.PHONE_LOOKING
        else:
            self._phone_look_start = None

        # ── Scenario 11: Phone call (moderate yaw + gaze off-center) ──
        if (detected_distraction == DistractionType.NONE and
                15.0 <= abs(yaw) <= 35.0 and abs(gaze_h - 0.5) > 0.15):
            if self._phone_call_start is None:
                self._phone_call_start = now
            elif now - self._phone_call_start >= self.phone_call_secs:
                detected_distraction = DistractionType.PHONE_CALL
        else:
            self._phone_call_start = None

        # ── Scenario 10: Talking to co-passenger (sustained yaw toward passenger) ──
        yaw_toward_passenger = yaw * self.copassenger_yaw_sign
        if (detected_distraction == DistractionType.NONE and
                yaw_toward_passenger > self.yaw_copassenger_threshold):
            if self._copassenger_start is None:
                self._copassenger_start = now
            elif now - self._copassenger_start >= self.copassenger_secs:
                detected_distraction = DistractionType.TALKING_COPASSENGER
        else:
            self._copassenger_start = None

        # ── Scenario 7: Long distraction (sustained looking away) ──
        if (detected_distraction == DistractionType.NONE and
                abs(yaw) > self.yaw_threshold):
            if self._looking_away_start is None:
                self._looking_away_start = now
            elif now - self._looking_away_start >= self.distraction_long_secs:
                detected_distraction = DistractionType.LOOKING_AWAY
        else:
            self._looking_away_start = None

        # ── Scenario 8: Repeated glances / VATS ──
        if detected_distraction == DistractionType.NONE:
            if abs(yaw) > self.yaw_glance_threshold:
                # We're currently looking away — check if this is a new glance
                if (not self._glance_timestamps or
                        now - self._glance_timestamps[-1] > 0.5):
                    self._glance_timestamps.append(now)

            # Clean old glances outside the window
            while (self._glance_timestamps and
                   now - self._glance_timestamps[0] > self.vats_window):
                self._glance_timestamps.popleft()

            if len(self._glance_timestamps) >= self.vats_count:
                detected_distraction = DistractionType.REPEATED_GLANCES

        # ── Scenario 14: Smoking (MAR oscillation pattern) ──
        if detected_distraction == DistractionType.NONE:
            if self._check_smoking_pattern(now):
                detected_distraction = DistractionType.SMOKING

        # ── Scenario 15: Eating/Drinking (head tilted back + mouth open) ──
        if (detected_distraction == DistractionType.NONE and
                pitch > 15.0 and mar > 0.5 and
                self.yawn_counter < self.mar_consec_frames):
            # Mouth open but not long enough to be yawning
            if self._eating_start is None:
                self._eating_start = now
            elif now - self._eating_start >= self.eating_secs:
                detected_distraction = DistractionType.EATING_DRINKING
        else:
            self._eating_start = None

        # ── Scenario 16: Unresponsive driver ──
        if detected_distraction == DistractionType.NONE:
            if self._check_unresponsive(now):
                detected_distraction = DistractionType.UNRESPONSIVE

        # ── Scenario 6: Head nodding (pitch distraction — catch-all) ──
        if (detected_distraction == DistractionType.NONE and
                abs(pitch) > self.pitch_threshold):
            detected_distraction = DistractionType.LOOKING_AWAY

        self.distraction_type = detected_distraction
        return self.state, self.distraction_type, self.yawning

    # ──────────────── Private detection helpers ────────────────

    def _check_no_face(self, now: float) -> DistractionType:
        """Check if no-face condition has persisted long enough."""
        if self._no_face_start is None:
            self._no_face_start = now
        if now - self._no_face_start >= self.no_face_secs:
            return DistractionType.NO_FACE
        return DistractionType.NONE

    def _check_face_obscured(self, now: float) -> DistractionType:
        """Check if face-obscured condition has persisted long enough."""
        if self._face_obscured_start is None:
            self._face_obscured_start = now
        if now - self._face_obscured_start >= self.face_obscured_secs:
            return DistractionType.FACE_OBSCURED
        return DistractionType.NONE

    def _check_smoking_pattern(self, now: float) -> bool:
        """Detect smoking-like MAR oscillation pattern.

        Smoking shows repeated short mouth openings (MAR 0.15-0.40)
        with a frequency of ~0.5-2Hz, distinct from yawning (wider,
        sustained) and talking (irregular, wider range).
        """
        if len(self._mar_history) < 20:
            return False

        # Check if MAR values are oscillating in the smoking range
        recent_mar = list(self._mar_history)
        recent_times = list(self._mar_time_history)

        # Time span of the history
        time_span = recent_times[-1] - recent_times[0]
        if time_span < 0.5:
            return False

        # Count zero-crossings around the midpoint (oscillation indicator)
        midpoint = 0.275  # midpoint of smoking MAR range (0.15-0.40)
        crossings = 0
        in_smoking_range = 0

        for i in range(1, len(recent_mar)):
            val = recent_mar[i]
            prev = recent_mar[i - 1]

            # Check if in smoking MAR range
            if 0.15 <= val <= 0.40:
                in_smoking_range += 1

            # Count crossings around midpoint
            if (prev < midpoint and val >= midpoint) or \
               (prev >= midpoint and val < midpoint):
                crossings += 1

        # Need most values in smoking range and enough oscillations
        smoking_ratio = in_smoking_range / len(recent_mar)
        oscillation_freq = crossings / time_span if time_span > 0 else 0

        if smoking_ratio > 0.6 and oscillation_freq >= 0.5:
            if self._smoking_start is None:
                self._smoking_start = now
            elif now - self._smoking_start >= self.smoking_secs:
                return True
        else:
            self._smoking_start = None

        return False

    def _check_unresponsive(self, now: float) -> bool:
        """Detect unresponsive driver (all metrics frozen for 15s+).

        Checks variance of recent metric snapshots. If variance is
        near zero across all channels for the configured duration,
        the driver may be incapacitated.
        """
        if len(self._metric_history) < 10:
            self._unresponsive_start = None
            return False

        recent = list(self._metric_history)

        # Check variance of each metric
        all_static = True
        for channel_idx in range(len(recent[0])):
            values = [m[channel_idx] for m in recent]
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)

            # Threshold for "static" — tiny movements are normal
            if variance > 0.001:
                all_static = False
                break

        if all_static:
            if self._unresponsive_start is None:
                self._unresponsive_start = now
            elif now - self._unresponsive_start >= self.unresponsive_secs:
                return True
        else:
            self._unresponsive_start = None

        return False

    @property
    def time_in_state(self) -> float:
        """Seconds elapsed since the last state transition."""
        return time.time() - self.last_state_change

    @property
    def is_distracted(self) -> bool:
        """Whether any distraction is currently active."""
        return self.distraction_type != DistractionType.NONE
