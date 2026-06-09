"""Automated scenario test runner for the Driver Fatigue Detection System.

Validates all 16 detection scenarios by feeding synthetic EAR/MAR/pitch/yaw/
gaze values into the DecisionEngine and verifying correct state transitions,
distraction types, and yawning detection.

No camera required — runs entirely with synthetic data.

Usage:
    python test_scenarios.py

Author: Driver Fatigue Detection System
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector.decision import DecisionEngine, DriverState, DistractionType


# ══════════════════════════════════════════════════════════════════════
#  Test Helpers
# ══════════════════════════════════════════════════════════════════════

class ScenarioTestRunner:
    """Runs all 16 detection scenarios and reports PASS/FAIL."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def _make_engine(self, **overrides) -> DecisionEngine:
        """Create a fresh DecisionEngine with fast timing for tests."""
        defaults = dict(
            ear_threshold=0.25,
            drowsy_frames=5,            # Faster for tests
            very_drowsy_frames=15,
            asleep_frames=25,
            mar_threshold=0.45,
            mar_consec_frames=3,
            pitch_threshold=20.0,
            yaw_threshold=30.0,
            perclos_threshold=0.15,
            yaw_glance_threshold=20.0,
            yaw_copassenger_threshold=25.0,
            distraction_long_secs=0.5,   # Fast for testing
            vats_count=3,
            vats_window=10.0,
            phone_look_secs=0.3,
            copassenger_secs=0.5,
            phone_call_secs=0.5,
            unresponsive_secs=0.5,
            smoking_secs=0.3,
            eating_secs=0.3,
            face_obscured_secs=0.3,
            no_face_secs=0.3,
            copassenger_side="right",
        )
        defaults.update(overrides)
        return DecisionEngine(**defaults)

    def _feed_frames(self, engine: DecisionEngine, count: int, **kwargs):
        """Feed N frames of identical data into the engine."""
        result = None
        for _ in range(count):
            result = engine.update(**kwargs)
            # Small sleep to allow time-based thresholds to trigger
            time.sleep(0.01)
        return result

    def _record(self, scenario_num: int, name: str, passed: bool,
                expected: str, actual: str):
        """Record a test result."""
        status = "PASS" if passed else "FAIL"
        self.results.append((scenario_num, name, status, expected, actual))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    # ══════════════════════════════════════════════════════════════════
    #  Scenario Tests
    # ══════════════════════════════════════════════════════════════════

    def test_01_drowsy(self):
        """Scenario 1: Microsleep — EAR below threshold for N frames."""
        engine = self._make_engine()
        result = self._feed_frames(
            engine, 10,
            ear=0.15, mar=0.0, pitch=0.0, yaw=0.0, perclos=0.0
        )
        state, dist, yawn = result
        passed = state == DriverState.DROWSY
        self._record(1, "Microsleep (EAR)", passed,
                     "DROWSY", state.name)

    def test_02_very_drowsy(self):
        """Scenario 2: Prolonged eye closure — VERY_DROWSY."""
        engine = self._make_engine()
        result = self._feed_frames(
            engine, 20,
            ear=0.10, mar=0.0, pitch=0.0, yaw=0.0, perclos=0.0
        )
        state, dist, yawn = result
        passed = state == DriverState.VERY_DROWSY
        self._record(2, "Prolonged closure (EAR)", passed,
                     "VERY_DROWSY", state.name)

    def test_03_asleep(self):
        """Scenario 3: Asleep — eyes shut for many frames."""
        engine = self._make_engine()
        result = self._feed_frames(
            engine, 30,
            ear=0.05, mar=0.0, pitch=0.0, yaw=0.0, perclos=0.0
        )
        state, dist, yawn = result
        passed = state == DriverState.ASLEEP
        self._record(3, "Asleep (EAR)", passed,
                     "ASLEEP", state.name)

    def test_04_perclos(self):
        """Scenario 4: PERCLOS fatigue — high PERCLOS triggers DROWSY."""
        engine = self._make_engine()
        result = self._feed_frames(
            engine, 5,
            ear=0.30, mar=0.0, pitch=0.0, yaw=0.0, perclos=0.20
        )
        state, dist, yawn = result
        passed = state == DriverState.DROWSY
        self._record(4, "PERCLOS fatigue", passed,
                     "DROWSY", state.name)

    def test_05_yawning(self):
        """Scenario 5: Yawning — high MAR for consecutive frames."""
        engine = self._make_engine()
        result = self._feed_frames(
            engine, 5,
            ear=0.30, mar=0.80, pitch=0.0, yaw=0.0, perclos=0.0
        )
        state, dist, yawn = result
        passed = yawn is True
        self._record(5, "Yawning (MAR)", passed,
                     "Yawning=True", f"Yawning={yawn}")

    def test_06_head_nodding(self):
        """Scenario 6: Head nodding — pitch exceeds threshold."""
        engine = self._make_engine()
        # Feed frames with enough time for threshold
        for _ in range(10):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=-30.0, yaw=0.0, perclos=0.0
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.LOOKING_AWAY
        self._record(6, "Head nodding (Pitch)", passed,
                     "LOOKING_AWAY", dist.name)

    def test_07_long_distraction(self):
        """Scenario 7: Long distraction — sustained yaw > 30° for 3s+."""
        engine = self._make_engine(distraction_long_secs=0.2)
        for _ in range(30):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=0.0, yaw=40.0, perclos=0.0,
                gaze_h=0.75, gaze_v=0.5
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.LOOKING_AWAY
        self._record(7, "Long distraction (Yaw)", passed,
                     "LOOKING_AWAY", dist.name)

    def test_08_repeated_glances(self):
        """Scenario 8: VATS — repeated short glances."""
        engine = self._make_engine()
        # Simulate 4 short glances with gaps
        for glance in range(4):
            # Look away briefly
            for _ in range(5):
                result = engine.update(
                    ear=0.30, mar=0.0, pitch=0.0, yaw=25.0, perclos=0.0,
                    gaze_h=0.75, gaze_v=0.5
                )
                time.sleep(0.01)
            # Look back
            for _ in range(10):
                result = engine.update(
                    ear=0.30, mar=0.0, pitch=0.0, yaw=0.0, perclos=0.0
                )
                time.sleep(0.01)
            time.sleep(0.2)  # Gap between glances

        # Check final state
        state, dist, yawn = result
        # After glances, the VATS counter should have registered enough
        # Check during a glance-away
        for _ in range(5):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=0.0, yaw=25.0, perclos=0.0,
                gaze_h=0.75, gaze_v=0.5
            )
        state, dist, yawn = result
        passed = dist == DistractionType.REPEATED_GLANCES
        self._record(8, "Repeated glances (VATS)", passed,
                     "REPEATED_GLANCES", dist.name)

    def test_09_phone_looking(self):
        """Scenario 9: Looking at phone — gaze down + pitch down."""
        engine = self._make_engine(phone_look_secs=0.2)
        for _ in range(30):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=-20.0, yaw=0.0, perclos=0.0,
                gaze_h=0.5, gaze_v=0.75
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.PHONE_LOOKING
        self._record(9, "Phone looking (Gaze)", passed,
                     "PHONE_LOOKING", dist.name)

    def test_10_copassenger(self):
        """Scenario 10: Talking to co-passenger — head turned right."""
        engine = self._make_engine(copassenger_secs=0.2)
        for _ in range(30):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=0.0, yaw=30.0, perclos=0.0
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.TALKING_COPASSENGER
        self._record(10, "Co-passenger (Yaw right)", passed,
                     "TALKING_COPASSENGER", dist.name)

    def test_11_phone_call(self):
        """Scenario 11: Phone call — moderate yaw + centered gaze + moderate roll."""
        engine = self._make_engine(phone_call_secs=0.2)
        for _ in range(30):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=0.0, yaw=20.0, perclos=0.0,
                gaze_h=0.5, gaze_v=0.5, roll=5.0
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.PHONE_CALL
        self._record(11, "Phone call (Yaw + Gaze)", passed,
                     "PHONE_CALL", dist.name)

    def test_12_no_face(self):
        """Scenario 12: No face detected for >2s."""
        engine = self._make_engine(no_face_secs=0.2)
        for _ in range(30):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=0.0, yaw=0.0, perclos=0.0,
                face_detected=False
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.NO_FACE
        self._record(12, "No face (2s+)", passed,
                     "NO_FACE", dist.name)

    def test_13_face_obscured(self):
        """Scenario 13: Face obscured — low confidence for >3s."""
        engine = self._make_engine(face_obscured_secs=0.2)
        for _ in range(30):
            result = engine.update(
                ear=0.30, mar=0.0, pitch=0.0, yaw=0.0, perclos=0.0,
                face_detected=True, face_confidence=0.3
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.FACE_OBSCURED
        self._record(13, "Face obscured (confidence)", passed,
                     "FACE_OBSCURED", dist.name)

    def test_14_smoking(self):
        """Scenario 14: Smoking — MAR oscillation pattern."""
        import math
        engine = self._make_engine(smoking_secs=0.1)
        phase = 0.0
        for i in range(200):
            # Oscillate MAR in smoking range (0.15 - 0.40)
            phase += 0.15
            mar = 0.275 + 0.12 * math.sin(phase * 5.0)
            result = engine.update(
                ear=0.30, mar=mar, pitch=0.0, yaw=0.0, perclos=0.0
            )
            time.sleep(0.005)
        state, dist, yawn = result
        passed = dist == DistractionType.SMOKING
        self._record(14, "Smoking (MAR oscillation)", passed,
                     "SMOKING", dist.name)

    def test_15_eating_drinking(self):
        """Scenario 15: Eating/Drinking — head tilted back + mouth open."""
        # Use high mar_consec_frames so yawn doesn't trigger before eating
        engine = self._make_engine(eating_secs=0.1, mar_consec_frames=100)
        for _ in range(50):
            result = engine.update(
                ear=0.30, mar=0.55, pitch=20.0, yaw=0.0, perclos=0.0
            )
            time.sleep(0.005)
        state, dist, yawn = result
        passed = dist == DistractionType.EATING_DRINKING
        self._record(15, "Eating/Drinking (tilt + MAR)", passed,
                     "EATING_DRINKING", dist.name)

    def test_16_unresponsive(self):
        """Scenario 16: Unresponsive — all metrics static for 15s+."""
        engine = self._make_engine(unresponsive_secs=0.3)
        # Feed identical frames for enough time
        for _ in range(50):
            result = engine.update(
                ear=0.30, mar=0.1, pitch=0.0, yaw=0.0, perclos=0.0,
                gaze_h=0.5, gaze_v=0.5
            )
            time.sleep(0.01)
        state, dist, yawn = result
        passed = dist == DistractionType.UNRESPONSIVE
        self._record(16, "Unresponsive (static 15s+)", passed,
                     "UNRESPONSIVE", dist.name)

    # ══════════════════════════════════════════════════════════════════
    #  Runner
    # ══════════════════════════════════════════════════════════════════

    def run_all(self):
        """Execute all scenario tests and print report."""
        print()
        print("=" * 60)
        print("  PRODUCTION SCENARIO TEST RUNNER")
        print("  Driver Fatigue & Distraction Detection System")
        print("=" * 60)
        print()
        print("Running 16 scenarios...\n")

        tests = [
            self.test_01_drowsy,
            self.test_02_very_drowsy,
            self.test_03_asleep,
            self.test_04_perclos,
            self.test_05_yawning,
            self.test_06_head_nodding,
            self.test_07_long_distraction,
            self.test_08_repeated_glances,
            self.test_09_phone_looking,
            self.test_10_copassenger,
            self.test_11_phone_call,
            self.test_12_no_face,
            self.test_13_face_obscured,
            self.test_14_smoking,
            self.test_15_eating_drinking,
            self.test_16_unresponsive,
        ]

        for test_fn in tests:
            test_fn()

        # Print report
        print()
        print("=" * 60)
        print("  SCENARIO TEST REPORT")
        print("=" * 60)

        for num, name, status, expected, actual in self.results:
            icon = "[PASS]" if status == "PASS" else "[FAIL]"
            line = f"  {icon} [{status}] {num:>2}. {name:<32s}"
            if status == "PASS":
                line += f" -> {actual}"
            else:
                line += f" -> Expected: {expected}, Got: {actual}"
            print(line)

        print("=" * 60)
        total = self.passed + self.failed
        if self.failed == 0:
            print(f"  [OK] {self.passed}/{total} scenarios PASSED")
        else:
            print(f"  [!!] {self.passed}/{total} passed, "
                  f"{self.failed} FAILED")
        print("=" * 60)

        return self.failed == 0


def main():
    """Run all scenario tests."""
    runner = ScenarioTestRunner()
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
