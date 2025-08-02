import cv2
import dlib
import numpy as np

# Load face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_head_pose(shape):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    size = (480, 640)
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch, yaw, roll = [float(angle) for angle in euler_angles]
    return pitch, yaw, roll

def detect_face_and_direction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return {
            "status": "No face detected",
            "violation": True,
            "violation_type": "no_face"
        }

    if len(faces) > 1:
        return {
            "status": "Multiple faces detected",
            "violation": True,
            "violation_type": "multiple_faces"
        }

    face = faces[0]
    shape = predictor(gray, face)
    pitch, yaw, roll = get_head_pose(shape)

    # You can tune these yaw thresholds
    if abs(yaw) > 20:
        return {
            "status": "Looking away",
            "violation": True,
            "violation_type": "looking_away"
        }

    return {
        "status": "Face detected",
        "violation": False,
        "violation_type": None
    }
