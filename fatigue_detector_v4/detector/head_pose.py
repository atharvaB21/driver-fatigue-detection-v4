"""Head pose estimation using OpenCV solvePnP.

Maps 6 canonical 3D facial points to their 2D landmark counterparts
using the Perspective-n-Point (PnP) algorithm to estimate the head's
pitch, yaw, and roll angles in real-time.
"""

import numpy as np
import cv2
import math


# 3D model points for a generic face (nose tip at origin).
# Coordinate system: X right, Y up, Z toward camera (face-centric).
# solvePnP handles the mapping to OpenCV camera coordinates.
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),       # Nose tip
    (0.0,   -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0,  170.0, -135.0),     # Right eye corner
    (-150.0,-150.0, -125.0),     # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)


def get_camera_matrix(frame_shape: tuple) -> tuple:
    """Build an approximate camera intrinsic matrix from frame dimensions.

    Uses the frame width as the focal length (a reasonable approximation
    for most webcams) and assumes no lens distortion.

    Parameters
    ----------
    frame_shape : tuple
        (height, width, ...) of the video frame.

    Returns
    -------
    tuple
        (camera_matrix, dist_coeffs) where camera_matrix is 3x3 and
        dist_coeffs is (4,1) zeros.
    """
    h, w = frame_shape[:2]
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)

    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """Convert a 3x3 rotation matrix to Euler angles (pitch, yaw, roll).

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix from cv2.Rodrigues.

    Returns
    -------
    tuple
        (pitch, yaw, roll) in degrees.
        - pitch: rotation around X-axis (nodding up/down)
        - yaw:   rotation around Y-axis (turning left/right)
        - roll:  rotation around Z-axis (tilting head)
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll)
    )


def estimate_head_pose(image_points: np.ndarray, frame_shape: tuple) -> tuple:
    """Estimate head pose from 6 facial landmark points using solvePnP.

    Parameters
    ----------
    image_points : np.ndarray
        6 facial landmark points in 2D, shape (6, 2), dtype float64.
        Order must match MODEL_POINTS_3D:
        [nose_tip, chin, left_eye_corner, right_eye_corner,
         left_mouth_corner, right_mouth_corner]
    frame_shape : tuple
        (height, width, ...) of the video frame.

    Returns
    -------
    tuple
        ((pitch, yaw, roll), rvec, tvec)
        - Angles in degrees
        - rvec, tvec: rotation/translation vectors for visualization
        - Returns ((0,0,0), None, None) on failure
    """
    camera_matrix, dist_coeffs = get_camera_matrix(frame_shape)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return (0.0, 0.0, 0.0), None, None

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler(R)

    # Correct pitch for the face-centric coordinate system.
    # The 3D model uses Y-up / Z-toward-camera, so solvePnP's raw
    # Euler decomposition includes a ~180° pitch offset.
    if pitch > 0:
        pitch = 180.0 - pitch
    else:
        pitch = -(180.0 + pitch)

    return (pitch, yaw, roll), rvec, tvec


def draw_pose_axes(frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                   camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                   length: float = 100.0) -> None:
    """Draw 3D coordinate axes on the frame to visualize head pose.

    Draws red (X), green (Y), and blue (Z) axes projected from the
    nose tip to show the head's orientation.

    Parameters
    ----------
    frame : np.ndarray
        The video frame to draw on (modified in-place).
    rvec : np.ndarray
        Rotation vector from solvePnP.
    tvec : np.ndarray
        Translation vector from solvePnP.
    camera_matrix : np.ndarray
        3x3 camera intrinsic matrix.
    dist_coeffs : np.ndarray
        Distortion coefficients.
    length : float
        Length of the axes in 3D units.
    """
    if rvec is None or tvec is None:
        return

    # Define axis endpoints in 3D
    axis_points = np.array([
        [length, 0,      0     ],   # X-axis (red)
        [0,      length, 0     ],   # Y-axis (green)
        [0,      0,      length]    # Z-axis (blue)
    ], dtype=np.float64)

    # Project 3D points to 2D image plane
    img_pts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    nose_2d, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 0.0)], dtype=np.float64),
        rvec, tvec, camera_matrix, dist_coeffs
    )

    origin = tuple(nose_2d[0].ravel().astype(int))

    # Draw axes: X=Red, Y=Green, Z=Blue
    cv2.line(frame, origin, tuple(img_pts[0].ravel().astype(int)), (0, 0, 255), 2)
    cv2.line(frame, origin, tuple(img_pts[1].ravel().astype(int)), (0, 255, 0), 2)
    cv2.line(frame, origin, tuple(img_pts[2].ravel().astype(int)), (255, 0, 0), 2)
