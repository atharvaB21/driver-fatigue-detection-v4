"""Driver Fatigue & Distraction Detection System — Main Entry Point.

Production-level driver monitoring system that detects 16 alert scenarios:
drowsiness, yawning, distraction (phone, co-passenger, looking away),
smoking, eating, unresponsive driver, and more.

Uses IP camera (RTSP) feed with MediaPipe facial landmarks, iris-based
gaze estimation, head pose via solvePnP, and Kalman-filtered smoothing.

Optimized for Intel NUC (Pentium N3700) with --low-power mode.

Usage:
    python main.py                      # Normal mode (default camera)
    python main.py --low-power          # Low-power mode for NUC/SBC
    python main.py --test-mode          # Enable keyboard scenario injection
    python main.py --source 1           # Use webcam at index 1
    python main.py --source video.mp4   # Use video file
    python main.py --process-width 240  # Custom processing resolution
    python main.py --frame-skip 3       # Process every 3rd frame

Controls:
    q     - Quit
    r     - Reset state machine
    l     - Toggle landmark display
    h     - Toggle HUD display
    s     - Take screenshot

Test Mode Controls (--test-mode):
    1     - Simulate Drowsy
    2     - Simulate Very Drowsy
    3     - Simulate Asleep
    4     - Simulate Yawning
    5     - Simulate Looking Away
    6     - Simulate Phone Looking
    7     - Simulate Co-passenger
    8     - Simulate Phone Call
    9     - Simulate Smoking
    0     - Simulate No Face

Author: Driver Fatigue Detection System
"""

import sys
import os
import time
import argparse
import math

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from detector.face import FaceDetector
from detector.ear import compute_ear
from detector.mar import mouth_aspect_ratio
from detector.head_pose import estimate_head_pose, get_camera_matrix, draw_pose_axes
from detector.perclos import PERCLOS
from detector.decision import DecisionEngine, DriverState, DistractionType
from detector.gaze import GazeEstimator
from detector.smoothing import MetricSmoother
from alerts.alerter import Alerter
from utils.logger import FatigueLogger


# ══════════════════════════════════════════════════════════════════════
#  Test Mode — Keyboard Scenario Simulator
# ══════════════════════════════════════════════════════════════════════

class ScenarioSimulator:
    """Manages keyboard-triggered scenario injection for testing.

    When active, overrides real sensor values with synthetic ones
    to simulate specific alert scenarios.
    """

    # Key -> (scenario_name, duration_seconds, overrides_dict)
    SCENARIOS = {
        ord('1'): ("Drowsy", 2.0, {"ear": 0.15}),
        ord('2'): ("Very Drowsy", 4.0, {"ear": 0.10}),
        ord('3'): ("Asleep", 6.0, {"ear": 0.05}),
        ord('4'): ("Yawning", 3.0, {"mar": 0.80}),
        ord('5'): ("Looking Away", 4.0, {"yaw": 40.0}),
        ord('6'): ("Phone Looking", 3.0, {"gaze_v": 0.80, "pitch": -25.0}),
        ord('7'): ("Co-passenger", 5.0, {"yaw": 35.0}),
        ord('8'): ("Phone Call", 5.0, {"yaw": 25.0, "gaze_h": 0.75}),
        ord('9'): ("Smoking", 6.0, {"mar_oscillate": True}),
        ord('0'): ("No Face", 5.0, {"no_face": True}),
    }

    def __init__(self):
        self.active_scenario = None
        self.scenario_name = ""
        self.overrides = {}
        self.start_time = 0.0
        self.duration = 0.0
        self._smoking_phase = 0.0

    def handle_key(self, key: int) -> bool:
        """Handle a keypress. Returns True if a scenario was triggered."""
        if key in self.SCENARIOS:
            name, duration, overrides = self.SCENARIOS[key]
            self.active_scenario = key
            self.scenario_name = name
            self.overrides = overrides
            self.start_time = time.time()
            self.duration = duration
            self._smoking_phase = 0.0
            print(f"[TEST] Simulating: {name} (duration: {duration}s)")
            return True
        return False

    @property
    def is_active(self) -> bool:
        """Whether a scenario simulation is currently running."""
        if self.active_scenario is None:
            return False
        if time.time() - self.start_time > self.duration:
            print(f"[TEST] Scenario '{self.scenario_name}' ended.")
            self.active_scenario = None
            self.overrides = {}
            return False
        return True

    def apply(self, ear: float, mar: float, pitch: float, yaw: float,
              gaze_h: float, gaze_v: float, face_detected: bool
              ) -> tuple:
        """Apply simulation overrides to real sensor values.

        Returns
        -------
        tuple
            (ear, mar, pitch, yaw, gaze_h, gaze_v, face_detected)
        """
        if not self.is_active:
            return ear, mar, pitch, yaw, gaze_h, gaze_v, face_detected

        o = self.overrides

        if o.get("no_face"):
            return ear, mar, pitch, yaw, gaze_h, gaze_v, False

        if o.get("mar_oscillate"):
            # Simulate smoking MAR oscillation
            self._smoking_phase += 0.15
            mar = 0.275 + 0.12 * math.sin(self._smoking_phase * 5.0)
            return ear, mar, pitch, yaw, gaze_h, gaze_v, face_detected

        ear = o.get("ear", ear)
        mar = o.get("mar", mar)
        pitch = o.get("pitch", pitch)
        yaw = o.get("yaw", yaw)
        gaze_h = o.get("gaze_h", gaze_h)
        gaze_v = o.get("gaze_v", gaze_v)

        return ear, mar, pitch, yaw, gaze_h, gaze_v, face_detected


# ══════════════════════════════════════════════════════════════════════
#  Application
# ══════════════════════════════════════════════════════════════════════

def check_prerequisites() -> bool:
    """Check that required files exist before starting."""
    # Generate sound files if missing
    beep_path = os.path.join(Config.SOUNDS_DIR, 'beep.wav')
    if not os.path.exists(beep_path):
        print("[INFO] Sound files not found. Generating...")
        try:
            from generate_sounds import generate_all
            generate_all()
            print("[INFO] Sound files generated successfully.")
        except Exception as e:
            print(f"[WARN] Could not generate sounds: {e}")
            print("[WARN] Audio alerts will be disabled.")

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Driver Fatigue & Distraction Detection System"
    )
    parser.add_argument(
        '--source', default=None,
        help='Camera source: webcam index (0,1,...), video file path, or '
             'RTSP URL. Default: uses config.py CAMERA_SOURCE'
    )
    parser.add_argument(
        '--no-audio', action='store_true',
        help='Disable audio alerts'
    )
    parser.add_argument(
        '--expression', action='store_true',
        help='Enable CNN expression detection (requires DeepFace)'
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='Display frame width (default: 640)'
    )

    # ── Low-power / NUC optimization flags ──
    parser.add_argument(
        '--low-power', action='store_true',
        help='Enable low-power mode for weak hardware (NUC, Raspberry Pi). '
             'Reduces processing resolution, skips frames, and disables '
             'expensive overlays for higher FPS.'
    )
    parser.add_argument(
        '--process-width', type=int, default=None,
        help='Processing frame width for MediaPipe inference. '
             'Lower = faster. Only used with --low-power or standalone. '
             'Default: 320 in low-power mode, same as display otherwise.'
    )
    parser.add_argument(
        '--frame-skip', type=int, default=None,
        help='Process every Nth frame with MediaPipe, reuse cached '
             'landmarks for skipped frames. Default: 3 in low-power mode.'
    )
    parser.add_argument(
        '--display-width', type=int, default=None,
        help='Display output width. In low-power mode this can be set '
             'independently of processing width. Default: 480 in '
             'low-power mode, 640 otherwise.'
    )

    # ── Test mode ──
    parser.add_argument(
        '--test-mode', action='store_true',
        help='Enable test mode: keyboard shortcuts (1-9,0) inject '
             'simulated scenarios to verify alerts.'
    )

    # ── RTSP options ──
    parser.add_argument(
        '--rtsp-transport', default=None,
        help='RTSP transport protocol: "tcp" or "udp". '
             'Default: uses config.py RTSP_TRANSPORT'
    )

    # ── Feature toggles ──
    parser.add_argument(
        '--no-head-pose', action='store_true',
        help='Disable head pose estimation (disables distraction detection).'
    )
    parser.add_argument(
        '--no-gaze', action='store_true',
        help='Disable iris-based gaze estimation.'
    )

    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()

    # ── Determine effective settings ──
    low_power = args.low_power or Config.LOW_POWER_MODE
    head_pose_enabled = Config.HEAD_POSE_ENABLED and not args.no_head_pose
    gaze_enabled = Config.GAZE_ENABLED and not args.no_gaze
    test_mode = args.test_mode

    print("=" * 60)
    print("  Driver Fatigue & Distraction Detection System")
    print("  Production Build — 16 Scenario Detection")
    if low_power:
        print("  ⚡ LOW-POWER MODE ACTIVE")
    if test_mode:
        print("  🧪 TEST MODE ACTIVE (keys 1-9, 0 to inject scenarios)")
    print("=" * 60)
    print()

    # ── Check prerequisites ──
    if not check_prerequisites():
        sys.exit(1)

    # ── Configuration ──
    camera_source = Config.CAMERA_SOURCE
    if args.source is not None:
        try:
            camera_source = int(args.source)
        except ValueError:
            camera_source = args.source

    # Resolve display and processing widths
    if low_power:
        display_width = args.display_width or args.width or 480
        process_width = args.process_width or Config.PROCESS_WIDTH  # 160
        frame_skip = args.frame_skip or Config.FRAME_SKIP           # 3
        det_confidence = Config.LOW_POWER_DETECTION_CONFIDENCE      # 0.35
        trk_confidence = Config.LOW_POWER_TRACKING_CONFIDENCE       # 0.35
        show_landmarks = Config.LOW_POWER_SHOW_LANDMARKS            # False
        show_hud = Config.LOW_POWER_SHOW_HUD                        # True
    else:
        display_width = args.width or Config.FRAME_WIDTH  # 640
        process_width = args.process_width or display_width
        frame_skip = args.frame_skip or 1  # No skipping
        det_confidence = 0.5
        trk_confidence = 0.5
        show_landmarks = Config.SHOW_LANDMARKS
        show_hud = Config.SHOW_HUD

    # ── Configure RTSP environment ──
    rtsp_transport = args.rtsp_transport or Config.RTSP_TRANSPORT
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{rtsp_transport}"
        "|fflags;nobuffer"
        "|flags;low_delay"
        "|framedrop;1"
    )

    # ── Initialize camera ──
    print(f"[INFO] Opening camera source: {camera_source}")

    if isinstance(camera_source, str) and (
        camera_source.startswith("rtsp://") or
        camera_source.startswith("http://")
    ):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(camera_source)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {camera_source}")
        sys.exit(1)

    # Set camera properties — request display resolution from camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(display_width * 3 / 4))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {actual_w}x{actual_h}")

    # ── Print optimization settings ──
    if low_power:
        print(f"[INFO] Display resolution:    {display_width}px wide")
        print(f"[INFO] Processing resolution: {process_width}px wide")
        print(f"[INFO] Frame skip:            process every {frame_skip} frame(s)")
        print(f"[INFO] Detection confidence:  {det_confidence}")
        print(f"[INFO] Landmarks drawing:     {'ON' if show_landmarks else 'OFF'}")
    else:
        print(f"[INFO] Resolution: {display_width}px wide")

    print(f"[INFO] Head pose estimation:  {'ON' if head_pose_enabled else 'OFF'}")
    print(f"[INFO] Gaze estimation:       {'ON' if gaze_enabled else 'OFF'}")
    print(f"[INFO] RTSP transport:        {rtsp_transport}")

    # ── Initialize face detector (MediaPipe — no model download needed) ──
    print("[INFO] Initializing MediaPipe Face Mesh...")
    face_detector = FaceDetector(
        min_detection_confidence=det_confidence,
        min_tracking_confidence=trk_confidence,
    )
    print("[INFO] Face detector ready.")

    # ── Initialize modules ──
    perclos_tracker = PERCLOS(window_size=Config.PERCLOS_WINDOW)

    decision_engine = DecisionEngine(
        ear_threshold=Config.EAR_THRESHOLD,
        drowsy_frames=Config.EAR_CONSEC_FRAMES_DROWSY,
        very_drowsy_frames=Config.EAR_CONSEC_FRAMES_VERY_DROWSY,
        asleep_frames=Config.EAR_CONSEC_FRAMES_ASLEEP,
        mar_threshold=Config.MAR_THRESHOLD,
        mar_consec_frames=Config.MAR_CONSEC_FRAMES,
        pitch_threshold=Config.HEAD_PITCH_THRESHOLD,
        yaw_threshold=Config.HEAD_YAW_THRESHOLD,
        perclos_threshold=Config.PERCLOS_THRESHOLD,
        yaw_glance_threshold=Config.HEAD_YAW_GLANCE_THRESHOLD,
        yaw_copassenger_threshold=Config.HEAD_YAW_COPASSENGER_THRESHOLD,
        distraction_long_secs=Config.DISTRACTION_LONG_THRESHOLD,
        vats_count=Config.DISTRACTION_VATS_COUNT,
        vats_window=Config.DISTRACTION_VATS_WINDOW,
        phone_look_secs=Config.PHONE_LOOK_THRESHOLD,
        copassenger_secs=Config.COPASSENGER_THRESHOLD,
        phone_call_secs=Config.PHONE_CALL_THRESHOLD,
        unresponsive_secs=Config.UNRESPONSIVE_THRESHOLD,
        smoking_secs=Config.SMOKING_DURATION,
        eating_secs=Config.EATING_DRINKING_DURATION,
        face_obscured_secs=Config.FACE_OBSCURED_THRESHOLD,
        no_face_secs=Config.NO_FACE_THRESHOLD,
        phone_call_roll_threshold=Config.PHONE_CALL_ROLL_THRESHOLD,
        copassenger_side=Config.COPASSENGER_SIDE,
    )

    alerter = Alerter(
        sounds_dir=Config.SOUNDS_DIR,
        tts_cooldown=Config.TTS_COOLDOWN,
        grace_period=Config.ALERT_GRACE_PERIOD,
    )

    logger = FatigueLogger(db_path=Config.DB_PATH)

    # ── Gaze estimator ──
    gaze_estimator = GazeEstimator() if gaze_enabled else None

    # ── Kalman filter smoother ──
    smoother = MetricSmoother(
        process_noise=Config.KALMAN_PROCESS_NOISE,
        measurement_noise=Config.KALMAN_MEASUREMENT_NOISE,
    )

    # ── Test mode simulator ──
    simulator = ScenarioSimulator() if test_mode else None

    # ── Optional CNN expression detector ──
    expression_detector = None
    if not low_power and (args.expression or Config.EXPRESSION_ENABLED):
        from expression.cnn_expression import ExpressionDetector
        expression_detector = ExpressionDetector(
            interval=Config.EXPRESSION_INTERVAL
        )
        if expression_detector.available:
            print("[INFO] CNN expression detection enabled.")
        else:
            print("[WARN] DeepFace not available. Expression detection disabled.")
            expression_detector = None
    elif low_power and (args.expression or Config.EXPRESSION_ENABLED):
        print("[INFO] CNN expression detection disabled in low-power mode.")

    # ── FPS tracking ──
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    # ── Frame skip state ──
    skip_counter = 0
    cached_landmarks_list = []   # Cached landmarks from last processed frame
    cached_ear = 0.3
    cached_mar = 0.0
    cached_pitch = 0.0
    cached_yaw = 0.0
    cached_roll = 0.0
    cached_gaze_h = 0.5
    cached_gaze_v = 0.5
    cached_rvec = None
    cached_tvec = None

    # ── RTSP reconnection state ──
    failed_reads = 0
    reconnect_attempt = 0

    # ── Calibration state ──
    yaw_offset = 0.0
    pitch_offset = 0.0
    roll_offset = 0.0
    calibration_frames = []       # list of (pitch, yaw, roll) for auto-calibration
    is_calibrated = False
    calibration_target_count = 60  # 60 frames of face detection for baseline

    print()
    print("[INFO] System running. Press 'q' to quit.")
    if test_mode:
        print("[INFO] Test mode: 1=Drowsy 2=VeryDrowsy 3=Asleep "
              "4=Yawn 5=LookAway 6=PhoneLook")
        print("[INFO]           7=Copassenger 8=PhoneCall "
              "9=Smoking 0=NoFace")
    print("[INFO] Controls: r=reset, c=calibrate, l=landmarks, h=HUD, s=screenshot")
    print()

    # ══════════════════════ MAIN LOOP ══════════════════════
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                failed_reads += 1
                logger.log_rtsp_event('FRAME_LOSS')

                print(f"[WARN] Frame lost ({failed_reads}/"
                      f"{Config.RTSP_MAX_FAILED_READS})")

                if failed_reads >= Config.RTSP_MAX_FAILED_READS:
                    reconnect_attempt += 1
                    delay = min(
                        Config.RTSP_RECONNECT_BASE_DELAY * (2 ** (reconnect_attempt - 1)),
                        Config.RTSP_RECONNECT_MAX_DELAY
                    )
                    print(f"[WARN] Reconnecting camera "
                          f"(attempt {reconnect_attempt}, "
                          f"delay {delay:.1f}s)...")
                    logger.log_rtsp_event('RECONNECT')

                    cap.release()
                    time.sleep(delay)

                    if isinstance(camera_source, str) and (
                        camera_source.startswith("rtsp://") or
                        camera_source.startswith("http://")
                    ):
                        cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                    else:
                        cap = cv2.VideoCapture(camera_source)

                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    failed_reads = 0

                    # Flush stale buffered frames
                    for _ in range(5):
                        cap.read()

                continue

            failed_reads = 0
            reconnect_attempt = 0  # Reset backoff on successful read

            # Resize frame to display width
            h_orig, w_orig = frame.shape[:2]
            if w_orig != display_width:
                aspect = h_orig / w_orig
                display_h = int(display_width * aspect)
                frame = cv2.resize(frame, (display_width, display_h),
                                   interpolation=cv2.INTER_NEAREST)
            else:
                display_h = h_orig

            # ── Determine if we should run MediaPipe this frame ──
            skip_counter += 1
            should_process = (skip_counter >= frame_skip)
            if should_process:
                skip_counter = 0

            # ── Face detection + landmark extraction ──
            if should_process:
                if process_width < display_width:
                    # Downscale for faster inference
                    p_aspect = display_h / display_width
                    p_h = int(process_width * p_aspect)
                    process_frame = cv2.resize(
                        frame, (process_width, p_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    face_landmarks_list = face_detector.detect_and_get_landmarks(
                        process_frame,
                        scale_to=(display_width, display_h)
                    )
                else:
                    face_landmarks_list = face_detector.detect_and_get_landmarks(
                        frame
                    )
                cached_landmarks_list = face_landmarks_list
            else:
                face_landmarks_list = cached_landmarks_list

            # Default values if no face detected
            ear = 0.3
            mar = 0.0
            pitch, yaw, roll = 0.0, 0.0, 0.0
            gaze_h, gaze_v = 0.5, 0.5
            perclos_val = perclos_tracker.value
            expression = ""
            rvec, tvec = None, None
            face_detected = len(face_landmarks_list) > 0
            face_confidence = 1.0 if face_detected else 0.0

            if face_detected:
                # Use the first detected face
                landmarks = face_landmarks_list[0]

                # Draw face rectangle from landmarks
                if show_landmarks:
                    face_detector.draw_face_rect(frame, landmarks, color=(0, 255, 0))
                    face_detector.draw_landmarks(frame, landmarks)

                if should_process:
                    # ── EAR calculation ──
                    left_eye = face_detector.get_left_eye(landmarks)
                    right_eye = face_detector.get_right_eye(landmarks)
                    ear = compute_ear(left_eye, right_eye)

                    # ── MAR calculation ──
                    mouth_inner = face_detector.get_mouth_inner(landmarks)
                    mar = mouth_aspect_ratio(mouth_inner)

                    # Draw eye and mouth contours (skip in low-power mode)
                    if show_landmarks:
                        left_hull = cv2.convexHull(left_eye)
                        right_hull = cv2.convexHull(right_eye)
                        cv2.drawContours(frame, [left_hull], -1, (0, 255, 255), 1)
                        cv2.drawContours(frame, [right_hull], -1, (0, 255, 255), 1)
                        mouth_hull = cv2.convexHull(mouth_inner)
                        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 255), 1)

                    # ── Head pose estimation ──
                    if head_pose_enabled:
                        pose_points = face_detector.get_head_pose_points(landmarks)
                        (pitch, yaw, roll), rvec, tvec = estimate_head_pose(
                            pose_points, frame.shape
                        )

                        if not is_calibrated:
                            calibration_frames.append((pitch, yaw, roll))
                            if len(calibration_frames) >= calibration_target_count:
                                pitch_offset = sum(f[0] for f in calibration_frames) / len(calibration_frames)
                                yaw_offset = sum(f[1] for f in calibration_frames) / len(calibration_frames)
                                roll_offset = sum(f[2] for f in calibration_frames) / len(calibration_frames)
                                is_calibrated = True
                                print(f"[INFO] Auto-calibration complete. Offsets - Pitch: {pitch_offset:.1f}, Yaw: {yaw_offset:.1f}, Roll: {roll_offset:.1f}")

                        # Apply offsets to align head pose relative to the driver looking straight
                        pitch -= pitch_offset
                        yaw -= yaw_offset
                        roll -= roll_offset

                    # ── Gaze estimation (iris-based, zero extra cost) ──
                    if gaze_estimator is not None:
                        gaze_h, gaze_v = gaze_estimator.estimate(landmarks)

                    # Cache computed values for skipped frames
                    cached_ear = ear
                    cached_mar = mar
                    cached_pitch = pitch
                    cached_yaw = yaw
                    cached_roll = roll
                    cached_gaze_h = gaze_h
                    cached_gaze_v = gaze_v
                    cached_rvec = rvec
                    cached_tvec = tvec
                else:
                    # Skipped frame: reuse all cached metrics (no recomputation)
                    ear = cached_ear
                    mar = cached_mar
                    pitch = cached_pitch
                    yaw = cached_yaw
                    roll = cached_roll
                    gaze_h = cached_gaze_h
                    gaze_v = cached_gaze_v
                    rvec = cached_rvec
                    tvec = cached_tvec

                # Draw pose axes if we got valid results
                if rvec is not None and not low_power:
                    cam_matrix, dist_coeffs = get_camera_matrix(frame.shape)
                    draw_pose_axes(frame, rvec, tvec, cam_matrix, dist_coeffs)

                # ── PERCLOS update ──
                perclos_val = perclos_tracker.update(ear, Config.EAR_THRESHOLD)

            # ── Apply Kalman smoothing ──
            smoothed = smoother.update(
                ear=ear, mar=mar, pitch=pitch, yaw=yaw,
                roll=roll, gaze_h=gaze_h, gaze_v=gaze_v
            )
            s_ear = smoothed['ear']
            s_mar = smoothed['mar']
            s_pitch = smoothed['pitch']
            s_yaw = smoothed['yaw']
            s_roll = smoothed['roll']
            s_gaze_h = smoothed['gaze_h']
            s_gaze_v = smoothed['gaze_v']

            # ── Test mode: override values ──
            if simulator is not None:
                s_ear, s_mar, s_pitch, s_yaw, s_gaze_h, s_gaze_v, face_detected = (
                    simulator.apply(
                        s_ear, s_mar, s_pitch, s_yaw,
                        s_gaze_h, s_gaze_v, face_detected
                    )
                )

            # ── Decision engine (all 16 scenarios) ──
            state, distraction_type, yawning = decision_engine.update(
                s_ear, s_mar, s_pitch, s_yaw, perclos_val,
                gaze_h=s_gaze_h, gaze_v=s_gaze_v,
                face_detected=face_detected,
                face_confidence=face_confidence,
                roll=s_roll,
            )

            # ── Optional CNN expression ──
            if expression_detector is not None and face_detected:
                expression_detector.update(frame)
                expression = expression_detector.get_emotion()

            # ── Logging ──
            dist_name = distraction_type.name
            is_distracted = distraction_type != DistractionType.NONE

            logger.log_state_change(
                state, s_ear, s_mar, perclos_val, s_pitch, s_yaw,
                is_distracted, yawning, expression,
                dist_name, s_gaze_h, s_gaze_v, face_confidence
            )
            logger.log_distraction_change(
                state, distraction_type,
                s_ear, s_mar, perclos_val, s_pitch, s_yaw,
                yawning, s_gaze_h, s_gaze_v, face_confidence
            )
            logger.log_snapshot(
                state, s_ear, s_mar, perclos_val, s_pitch, s_yaw,
                is_distracted, yawning, expression,
                interval=Config.LOG_SNAPSHOT_INTERVAL,
                distraction_type=dist_name,
                gaze_h=s_gaze_h, gaze_v=s_gaze_v,
                face_confidence=face_confidence
            )

            # ── Alerts ──
            if not args.no_audio:
                alerter.play_alert(state, distraction_type)

            # ── Visual overlays ──
            if show_hud:
                alerter.draw_overlay(
                    frame, state, distraction_type, yawning,
                    s_ear, s_mar, perclos_val, s_pitch, s_yaw, fps,
                    gaze_h=s_gaze_h, gaze_v=s_gaze_v, roll=s_roll
                )

            # ── No face warning (when face_detected is False and no HUD) ──
            if not face_detected and not show_hud:
                cv2.putText(
                    frame, "NO FACE DETECTED",
                    (frame.shape[1] // 2 - 130, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2
                )

            # ── Expression label ──
            if expression and expression != "N/A":
                cv2.putText(
                    frame, f"Expr: {expression}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 200, 0), 2
                )

            # ── Test mode indicator ──
            if simulator is not None and simulator.is_active:
                elapsed = time.time() - simulator.start_time
                remaining = max(0, simulator.duration - elapsed)
                cv2.putText(
                    frame,
                    f"[TEST] {simulator.scenario_name} ({remaining:.1f}s)",
                    (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

            # ── Calibration Overlay ──
            if not is_calibrated and face_detected:
                cv2.putText(
                    frame, f"CALIBRATING HEAD POSE ({len(calibration_frames)}/{calibration_target_count})",
                    (10, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 140, 255), 2  # Orange color
                )

            # ── FPS calculation ──
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # ── Display ──
            cv2.imshow("Driver Fatigue Monitor", frame)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Quit requested.")
                break
            elif key == ord('r'):
                decision_engine.reset()
                perclos_tracker.reset()
                smoother.reset()
                print("[INFO] State machine reset.")
            elif key == ord('c'):
                calibration_frames = []
                is_calibrated = False
                pitch_offset = 0.0
                yaw_offset = 0.0
                roll_offset = 0.0
                print("[INFO] Re-calibrating head pose. Please look straight at the road...")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"[INFO] Landmarks {'ON' if show_landmarks else 'OFF'}")
            elif key == ord('h'):
                show_hud = not show_hud
                print(f"[INFO] HUD {'ON' if show_hud else 'OFF'}")
            elif key == ord('s'):
                screenshot_path = os.path.join(
                    Config.BASE_DIR,
                    f"screenshot_{int(time.time())}.png"
                )
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot saved: {screenshot_path}")
            elif simulator is not None:
                simulator.handle_key(key)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # ── Cleanup ──
        print("[INFO] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        alerter.cleanup()
        face_detector.close()

        # Print session summary
        summary = logger.get_session_summary()
        if summary and summary[0] > 0:
            print()
            print("=" * 60)
            print("  SESSION SUMMARY")
            print("=" * 60)
            print(f"  Total events logged:  {summary[0]}")
            print(f"  Drowsy alerts:        {summary[1]}")
            print(f"  Very drowsy alerts:   {summary[2]}")
            print(f"  Asleep alerts:        {summary[3]}")
            print(f"  Distraction events:   {summary[4]}")
            print(f"  Yawn events:          {summary[5]}")
            print(f"  Average EAR:          {summary[6]:.3f}" if summary[6] else "")
            print(f"  Average PERCLOS:      {summary[7]:.3f}" if summary[7] else "")
            print(f"  Session start:        {summary[8]}")
            print(f"  Session end:          {summary[9]}")
            print("-" * 60)
            print(f"  RTSP reconnections:   {logger.reconnect_count}")
            print(f"  Frame losses:         {logger.frame_loss_count}")
            print("=" * 60)

            # Distraction breakdown
            dist_summary = logger.get_distraction_summary()
            if dist_summary:
                print("  DISTRACTION BREAKDOWN:")
                for dtype, count in dist_summary:
                    print(f"    {dtype:.<30s} {count}")
                print("=" * 60)

            print(f"  Data saved to: {Config.DB_PATH}")
            print(f"  View dashboard: streamlit run dashboard/app.py")
            print("=" * 60)

        print("[INFO] Done.")


if __name__ == '__main__':
    main()
