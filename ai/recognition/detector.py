"""
Face Detection Module.
Detects faces in camera frames and images.
This is the first step in the recognition pipeline.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


class FaceDetector:
    """
    Detects and analyses faces in images.
    Uses InsightFace buffalo_l model.
    """

    def __init__(self):
        """
        Initialize the face detection model.
        Downloads model on first run automatically.
        """
        print("Initializing face detection model...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face detection model ready.")

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect all faces in a given image frame.

        Args:
            frame: BGR image array from OpenCV

        Returns:
            List of detected face objects.
            Each face contains:
            - bbox: bounding box coordinates
            - embedding: 512-dimension face vector
            - det_score: detection confidence
        """
        if frame is None:
            return []

        faces = self.app.get(frame)
        return faces

    def detect_largest_face(self, frame: np.ndarray):
        """
        Returns the largest detected face in the frame.
        Useful for single-person enrollment and recognition.

        Args:
            frame: BGR image array from OpenCV

        Returns:
            Single face object or None if no face detected
        """
        faces = self.detect(frame)

        if not faces:
            return None

        if len(faces) == 1:
            return faces[0]

        # Return largest face by bounding box area
        largest = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )
        return largest

    def draw_detection(
        self,
        frame: np.ndarray,
        face,
        label: str = "",
        confidence: float = None,
        color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw bounding box and label on frame.

        Args:
            frame:      BGR image array
            face:       detected face object
            label:      name or text to display
            confidence: match confidence score
            color:      BGR color tuple

        Returns:
            Frame with detection drawn on it
        """
        frame_copy = frame.copy()

        x1, y1, x2, y2 = [int(c) for c in face.bbox]

        # Draw bounding box
        cv2.rectangle(
            frame_copy,
            (x1, y1), (x2, y2),
            color, 2
        )

        # Build display text
        if label:
            display = label
            if confidence is not None:
                display += f" ({confidence:.1%})"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(
                display,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 2
            )
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color, -1
            )

            # Text
            cv2.putText(
                frame_copy, display,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2
            )

        return frame_copy

    def is_face_quality_acceptable(
        self,
        face,
        min_confidence: float = 0.7,
        min_size: int = 80
    ) -> tuple:
        """
        Check if detected face meets quality requirements
        for enrollment or recognition.

        Args:
            face:           detected face object
            min_confidence: minimum detection confidence
            min_size:       minimum face width in pixels

        Returns:
            Tuple of (is_acceptable, reason)
        """
        if face is None:
            return False, "No face detected"

        # Check detection confidence
        if face.det_score < min_confidence:
            return False, f"Low confidence: {face.det_score:.2f}"

        # Check face size
        x1, y1, x2, y2 = face.bbox
        face_width = x2 - x1

        if face_width < min_size:
            return False, f"Face too small: {int(face_width)}px"

        return True, "Acceptable"