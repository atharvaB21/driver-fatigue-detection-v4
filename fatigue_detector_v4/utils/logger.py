"""SQLite event logger for fatigue detection events.

Logs state changes, distraction events, and periodic snapshots to an
SQLite database for post-session analysis and dashboard visualization.

Extended schema includes distraction types, gaze data, and face
confidence for the 16-scenario production system.
"""

import sqlite3
import os
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Optional


class FatigueLogger:
    """Thread-safe SQLite logger for driver fatigue events.

    Creates a new connection for each operation to ensure thread safety.
    Logs two types of events:
    - STATE_CHANGE: recorded when the driver's state transitions
    - SNAPSHOT: periodic captures of all metrics (default every 30s)
    - DISTRACTION: recorded when a distraction type changes

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

        # Ensure the directory exists
        db_dir = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(db_dir, exist_ok=True)

        self._create_table()
        self.last_snapshot_time = time.time()
        self.last_state: Optional[str] = None
        self.last_distraction: Optional[str] = None

        # RTSP health tracking
        self.reconnect_count = 0
        self.frame_loss_count = 0

    @contextmanager
    def _get_connection(self):
        """Context manager for thread-safe database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _create_table(self) -> None:
        """Create the events table if it doesn't exist.

        Schema includes all production metrics: distraction type,
        gaze direction, and face confidence.
        """
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    ear REAL,
                    mar REAL,
                    perclos REAL,
                    pitch REAL,
                    yaw REAL,
                    distracted INTEGER DEFAULT 0,
                    yawning INTEGER DEFAULT 0,
                    expression TEXT DEFAULT '',
                    distraction_type TEXT DEFAULT 'NONE',
                    gaze_h REAL DEFAULT 0.5,
                    gaze_v REAL DEFAULT 0.5,
                    face_confidence REAL DEFAULT 1.0
                )
            ''')

            # Migrate existing tables: add new columns if they don't exist
            # This is safe to call repeatedly — SQLite ignores if column exists
            for col_def in [
                ("distraction_type", "TEXT DEFAULT 'NONE'"),
                ("gaze_h", "REAL DEFAULT 0.5"),
                ("gaze_v", "REAL DEFAULT 0.5"),
                ("face_confidence", "REAL DEFAULT 1.0"),
            ]:
                try:
                    conn.execute(
                        f"ALTER TABLE events ADD COLUMN {col_def[0]} {col_def[1]}"
                    )
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

    def log_event(self, event_type: str, state: str, ear: float,
                  mar: float, perclos: float, pitch: float, yaw: float,
                  distracted: bool, yawning: bool,
                  expression: str = '',
                  distraction_type: str = 'NONE',
                  gaze_h: float = 0.5, gaze_v: float = 0.5,
                  face_confidence: float = 1.0) -> None:
        """Log a single event to the database.

        Parameters
        ----------
        event_type : str
            Type of event ('STATE_CHANGE', 'SNAPSHOT', 'DISTRACTION', etc.)
        state : str
            Current driver state name.
        ear, mar, perclos, pitch, yaw : float
            Current metric values.
        distracted, yawning : bool
            Current flags.
        expression : str
            Detected facial expression (from CNN, if enabled).
        distraction_type : str
            Name of the DistractionType enum value.
        gaze_h, gaze_v : float
            Horizontal and vertical gaze ratios.
        face_confidence : float
            Face detection confidence (0.0 to 1.0).
        """
        with self._get_connection() as conn:
            conn.execute(
                '''INSERT INTO events
                   (timestamp, event_type, state, ear, mar, perclos,
                    pitch, yaw, distracted, yawning, expression,
                    distraction_type, gaze_h, gaze_v, face_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    datetime.now().isoformat(),
                    event_type, state,
                    ear, mar, perclos, pitch, yaw,
                    int(distracted), int(yawning), expression,
                    distraction_type, gaze_h, gaze_v, face_confidence
                )
            )

    def log_state_change(self, state, ear: float, mar: float,
                         perclos: float, pitch: float, yaw: float,
                         distracted: bool, yawning: bool,
                         expression: str = '',
                         distraction_type: str = 'NONE',
                         gaze_h: float = 0.5, gaze_v: float = 0.5,
                         face_confidence: float = 1.0) -> None:
        """Log a state change event (only if state actually changed).

        Parameters
        ----------
        state : DriverState or str
            The new driver state.
        """
        state_name = state.name if hasattr(state, 'name') else str(state)

        if state_name != self.last_state:
            self.log_event(
                'STATE_CHANGE', state_name,
                ear, mar, perclos, pitch, yaw,
                distracted, yawning, expression,
                distraction_type, gaze_h, gaze_v, face_confidence
            )
            self.last_state = state_name

    def log_distraction_change(self, state, distraction_type,
                               ear: float, mar: float, perclos: float,
                               pitch: float, yaw: float,
                               yawning: bool,
                               gaze_h: float = 0.5, gaze_v: float = 0.5,
                               face_confidence: float = 1.0) -> None:
        """Log a distraction type change event.

        Parameters
        ----------
        state : DriverState or str
            Current driver state.
        distraction_type : DistractionType or str
            The new distraction type.
        """
        state_name = state.name if hasattr(state, 'name') else str(state)
        dist_name = (distraction_type.name
                     if hasattr(distraction_type, 'name')
                     else str(distraction_type))

        if dist_name != self.last_distraction:
            distracted = dist_name != 'NONE'
            self.log_event(
                'DISTRACTION', state_name,
                ear, mar, perclos, pitch, yaw,
                distracted, yawning, '',
                dist_name, gaze_h, gaze_v, face_confidence
            )
            self.last_distraction = dist_name

    def log_snapshot(self, state, ear: float, mar: float,
                     perclos: float, pitch: float, yaw: float,
                     distracted: bool, yawning: bool,
                     expression: str = '',
                     interval: float = 30.0,
                     distraction_type: str = 'NONE',
                     gaze_h: float = 0.5, gaze_v: float = 0.5,
                     face_confidence: float = 1.0) -> None:
        """Log a periodic snapshot (rate-limited by interval).

        Parameters
        ----------
        interval : float
            Minimum seconds between snapshots. Default 30.
        """
        now = time.time()
        if now - self.last_snapshot_time >= interval:
            state_name = state.name if hasattr(state, 'name') else str(state)
            self.log_event(
                'SNAPSHOT', state_name,
                ear, mar, perclos, pitch, yaw,
                distracted, yawning, expression,
                distraction_type, gaze_h, gaze_v, face_confidence
            )
            self.last_snapshot_time = now

    def log_rtsp_event(self, event: str) -> None:
        """Log RTSP health event (reconnection, frame loss).

        Parameters
        ----------
        event : str
            Event description (e.g., 'RECONNECT', 'FRAME_LOSS').
        """
        if event == 'RECONNECT':
            self.reconnect_count += 1
        elif event == 'FRAME_LOSS':
            self.frame_loss_count += 1

        with self._get_connection() as conn:
            conn.execute(
                '''INSERT INTO events
                   (timestamp, event_type, state, ear, mar, perclos,
                    pitch, yaw, distracted, yawning, expression,
                    distraction_type, gaze_h, gaze_v, face_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    datetime.now().isoformat(),
                    f'RTSP_{event}', 'N/A',
                    0, 0, 0, 0, 0,
                    0, 0, '',
                    'NONE', 0.5, 0.5, 0.0
                )
            )

    def get_recent_events(self, limit: int = 100) -> list:
        """Retrieve the most recent events from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM events ORDER BY id DESC LIMIT ?',
                (limit,)
            )
            return cursor.fetchall()

    def get_session_summary(self) -> tuple:
        """Get aggregate statistics for the current session."""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT
                    COUNT(*) as total_events,
                    SUM(CASE WHEN state = 'DROWSY' THEN 1 ELSE 0 END) as drowsy_count,
                    SUM(CASE WHEN state = 'VERY_DROWSY' THEN 1 ELSE 0 END) as very_drowsy_count,
                    SUM(CASE WHEN state = 'ASLEEP' THEN 1 ELSE 0 END) as asleep_count,
                    SUM(distracted) as distraction_count,
                    SUM(yawning) as yawn_count,
                    AVG(ear) as avg_ear,
                    AVG(perclos) as avg_perclos,
                    MIN(timestamp) as session_start,
                    MAX(timestamp) as session_end
                FROM events
            ''')
            return cursor.fetchone()

    def get_distraction_summary(self) -> list:
        """Get breakdown of distraction events by type.

        Returns
        -------
        list
            List of (distraction_type, count) tuples.
        """
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT distraction_type, COUNT(*) as count
                FROM events
                WHERE distraction_type != 'NONE'
                  AND event_type = 'DISTRACTION'
                GROUP BY distraction_type
                ORDER BY count DESC
            ''')
            return cursor.fetchall()
