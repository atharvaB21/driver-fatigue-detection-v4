"""Face detection and landmark extraction using MediaPipe Face Mesh.

Replaces dlib with MediaPipe for hassle-free installation (no C++ compiler
required). MediaPipe Face Mesh provides 478 facial landmarks; this module
maps the relevant ones to the same roles as dlib's 68-point model for
EAR, MAR, head pose, etc.
"""

import numpy as np
import cv2
import os
import urllib.request

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] MediaPipe not installed. Face detection disabled.")

# ─────────────────────────────────────────────────────────────────────
#  MediaPipe Face Mesh landmark indices that correspond to dlib's 68
# ─────────────────────────────────────────────────────────────────────

# Left eye (6 points — matches dlib 36-41 layout)
# P1(left corner) P2(upper-left) P3(upper-right) P4(right corner) P5(lower-right) P6(lower-left)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Right eye (6 points — matches dlib 42-47 layout)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Inner mouth (8 points — matches dlib 60-67 layout)
# Top: 78(left), 81(upper-left), 13(top-center), 311(upper-right), 308(right)
# Bottom: 402(lower-right), 14(bottom-center), 178(lower-left)
MOUTH_INNER_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

# Outer mouth (for drawing, not used in MAR calculation)
MOUTH_OUTER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375]

# Head pose estimation — 6 canonical points
NOSE_TIP_INDEX = 1          # Nose tip
CHIN_INDEX = 152            # Chin
LEFT_EYE_CORNER_INDEX = 33  # Left eye inner corner
RIGHT_EYE_CORNER_INDEX = 263  # Right eye inner corner
LEFT_MOUTH_CORNER_INDEX = 61  # Left mouth corner
RIGHT_MOUTH_CORNER_INDEX = 291  # Right mouth corner

# All face contour for drawing
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                     288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                     150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                     67, 109, 10]


class FaceDetector:
    """Face detector and landmark extractor using MediaPipe Face Mesh.

    Parameters
    ----------
    model_path : str
        Path to the face landmarker model. If empty, defaults to Config.MODEL_PATH.
    use_cnn : bool
        Ignored (kept for API compatibility).
    max_faces : int
        Maximum number of faces to detect.
    min_detection_confidence : float
        Minimum confidence for face detection.
    min_tracking_confidence : float
        Minimum confidence for landmark tracking.
    """

    def __init__(self, model_path: str = "", use_cnn: bool = False,
                 max_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is not installed. "
                "Install with: pip install mediapipe"
            )

        # Dynamic fallback to config.py if no model_path is provided
        if not model_path:
            try:
                from config import Config
                model_path = Config.MODEL_PATH
            except ImportError:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        # Download model if not found
        if not os.path.exists(model_path):
            print(f"[INFO] MediaPipe model not found. Downloading to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("[INFO] Model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download MediaPipe model from {url}: {e}")

        # Initialize the FaceLandmarker Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self._last_landmarks = None
        self._last_face_rect = None

    def detect_and_get_landmarks(self, frame: np.ndarray,
                                  scale_to: tuple = None):
        """Detect faces and extract landmarks in one step.

        MediaPipe works on RGB frames and returns normalized landmarks.
        This method converts them to pixel coordinates.

        Parameters
        ----------
        frame : np.ndarray
            BGR video frame from OpenCV (may be a lower-resolution
            processing frame in low-power mode).
        scale_to : tuple, optional
            If provided as (target_width, target_height), the returned
            landmark pixel coordinates are scaled to that resolution
            instead of the input frame's resolution. This allows running
            inference on a small frame but getting landmarks in the
            display frame's coordinate space.

        Returns
        -------
        list of np.ndarray
            List of landmark arrays, one per detected face.
            Each array has shape (478, 2) with pixel coordinates.
            Returns empty list if no faces detected.
        """
        h, w = frame.shape[:2]
        # Determine output coordinate space
        if scale_to is not None:
            out_w, out_h = scale_to
        else:
            out_w, out_h = w, h

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Process detection
        results = self.detector.detect(mp_image)

        faces = []
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Vectorized conversion: extract all x,y into numpy arrays at once
                # This is much faster than a Python for-loop over 478 landmarks
                coords = np.empty((len(face_landmarks), 2), dtype=np.float32)
                for i, lm in enumerate(face_landmarks):
                    coords[i, 0] = lm.x
                    coords[i, 1] = lm.y
                coords[:, 0] *= out_w
                coords[:, 1] *= out_h
                landmarks = coords.astype(np.int32)
                faces.append(landmarks)

        self._last_landmarks = faces
        return faces

    def detect_faces(self, gray: np.ndarray):
        """Compatibility method — detects faces and returns a list.

        Note: MediaPipe processes detection + landmarks together.
        This method returns a list of face indices (not dlib rectangles).

        Parameters
        ----------
        gray : np.ndarray
            Grayscale frame (will be converted to RGB internally).

        Returns
        -------
        list
            List of detected face indices (0, 1, ...).
        """
        return list(range(len(self._last_landmarks or [])))

    def get_landmarks(self, gray: np.ndarray, face_idx: int = 0) -> np.ndarray:
        """Get landmarks for a specific face.

        Parameters
        ----------
        gray : np.ndarray
            Ignored (landmarks already extracted).
        face_idx : int
            Index into the cached landmarks list.

        Returns
        -------
        np.ndarray
            Array of shape (478, 2) with pixel coordinates.
        """
        if self._last_landmarks and face_idx < len(self._last_landmarks):
            return self._last_landmarks[face_idx]
        return np.array([])

    # ──────────── Landmark subset extractors ────────────

    def get_left_eye(self, landmarks: np.ndarray) -> np.ndarray:
        """Get 6 left eye landmark points."""
        return landmarks[LEFT_EYE_INDICES]

    def get_right_eye(self, landmarks: np.ndarray) -> np.ndarray:
        """Get 6 right eye landmark points."""
        return landmarks[RIGHT_EYE_INDICES]

    def get_mouth_outer(self, landmarks: np.ndarray) -> np.ndarray:
        """Get outer mouth landmark points."""
        return landmarks[MOUTH_OUTER_INDICES]

    def get_mouth_inner(self, landmarks: np.ndarray) -> np.ndarray:
        """Get 8 inner mouth landmark points for MAR calculation."""
        return landmarks[MOUTH_INNER_INDICES]

    def get_head_pose_points(self, landmarks: np.ndarray) -> np.ndarray:
        """Get the 6 points needed for head pose estimation.

        Returns them in the same order as the 3D model points in
        head_pose.py: [nose_tip, chin, left_eye, right_eye,
        left_mouth, right_mouth].
        """
        indices = [
            NOSE_TIP_INDEX, CHIN_INDEX,
            LEFT_EYE_CORNER_INDEX, RIGHT_EYE_CORNER_INDEX,
            LEFT_MOUTH_CORNER_INDEX, RIGHT_MOUTH_CORNER_INDEX
        ]
        return landmarks[indices].astype(np.float64)

    # ──────────── Drawing utilities ────────────

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray,
                       color: tuple = (0, 255, 0), radius: int = 1,
                       indices: list = None) -> None:
        """Draw landmark points on the frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on (modified in-place).
        landmarks : np.ndarray
            Full 478-point landmark array.
        color : tuple
            BGR color.
        radius : int
            Point radius.
        indices : list, optional
            If given, only draw these specific landmark indices.
            If None, draws key facial landmarks (not all 478).
        """
        if indices is None:
            # Draw a sensible subset: eyes, mouth, nose, brows
            indices = (LEFT_EYE_INDICES + RIGHT_EYE_INDICES +
                       MOUTH_INNER_INDICES + [NOSE_TIP_INDEX, CHIN_INDEX])

        for idx in indices:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)

    @staticmethod
    def draw_face_rect(frame: np.ndarray, landmarks: np.ndarray,
                       color: tuple = (0, 255, 0), thickness: int = 2) -> None:
        """Draw a bounding rectangle around the face.

        Computes the bounding box from the face oval landmarks.
        """
        if len(landmarks) == 0:
            return

        # Use face oval indices if available, otherwise use all
        try:
            oval_pts = landmarks[FACE_OVAL_INDICES]
        except IndexError:
            oval_pts = landmarks

        x_min = int(oval_pts[:, 0].min())
        y_min = int(oval_pts[:, 1].min())
        x_max = int(oval_pts[:, 0].max())
        y_max = int(oval_pts[:, 1].max())

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
