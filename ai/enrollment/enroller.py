"""
Student Face Enrollment Module.

Handles the complete enrollment pipeline:
1. Capture multiple face images from camera
2. Validate face quality for each capture
3. Generate face embeddings
4. Store embeddings in database
5. Log enrollment in audit trail

Privacy by Design:
Raw face images are NEVER stored.
Only the mathematical embedding vector
is persisted to the database.
"""

import cv2
import numpy as np
import json
from datetime import datetime
from sqlalchemy.orm import Session

from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder
from database.models import Student, FaceEmbedding, AuditLog
from database.connection import get_local_db


class StudentEnroller:
    """
    Manages the complete student enrollment process.
    """

    def __init__(
        self,
        required_captures: int = 5,
        capture_delay: float = 1.0
    ):
        """
        Initialize the enroller.

        Args:
            required_captures: number of face images to capture
                               More captures = better accuracy
                               5 is the recommended minimum

            capture_delay:     seconds between captures
                               Gives student time to adjust pose
        """
        self.detector          = FaceDetector()
        self.embedder          = FaceEmbedder()
        self.required_captures = required_captures
        self.capture_delay     = capture_delay

    def enroll_from_camera(
        self,
        student_id: str,
        student_name: str,
        db: Session
    ) -> dict:
        """
        Capture face images from webcam and enroll student.

        Opens the laptop camera, guides the student through
        multiple capture positions, generates embeddings,
        and stores them in the database.

        Args:
            student_id:   UUID of student in database
            student_name: Full name for display during capture
            db:           Database session

        Returns:
            Dict with enrollment result:
            - success:   True if enrollment completed
            - captures:  number of successful captures
            - message:   result description
        """
        print(f"\nStarting enrollment for: {student_name}")
        print(f"Required captures: {self.required_captures}")
        print("Press SPACE to capture. Press Q to quit.\n")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return {
                'success': False,
                'captures': 0,
                'message': 'Cannot access camera'
            }

        embeddings_collected = []
        capture_count        = 0
        last_capture_time    = 0

        try:
            while capture_count < self.required_captures:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror the frame for natural interaction
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # Detect face in current frame
                face = self.detector.detect_largest_face(frame)

                if face is not None:
                    is_ok, reason = self.detector.is_face_quality_acceptable(face)

                    if is_ok:
                        display_frame = self.detector.draw_detection(
                            display_frame, face,
                            label=f"Ready — {student_name}",
                            color=(0, 255, 0)
                        )
                        instruction = "SPACE to capture"
                        color = (0, 255, 0)
                    else:
                        display_frame = self.detector.draw_detection(
                            display_frame, face,
                            label=reason,
                            color=(0, 165, 255)
                        )
                        instruction = reason
                        color = (0, 165, 255)
                else:
                    instruction = "No face detected — position face in frame"
                    color = (0, 0, 255)

                # Display progress
                cv2.putText(
                    display_frame,
                    f"Captures: {capture_count}/{self.required_captures}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )
                cv2.putText(
                    display_frame,
                    instruction,
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )
                cv2.putText(
                    display_frame,
                    "Q — quit enrollment",
                    (10, display_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1
                )

                cv2.imshow(f"Almanac AI — Enrolling {student_name}", display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("Enrollment cancelled by user.")
                    break

                if key == ord(' ') and face is not None:
                    is_ok, reason = self.detector.is_face_quality_acceptable(face)

                    if is_ok:
                        embedding = self.embedder.extract(face)
                        embeddings_collected.append(embedding)
                        capture_count += 1
                        print(f"  Capture {capture_count}/{self.required_captures} saved.")

                        # Flash green to confirm capture
                        confirm_frame = frame.copy()
                        cv2.rectangle(
                            confirm_frame,
                            (0, 0),
                            (confirm_frame.shape[1], confirm_frame.shape[0]),
                            (0, 255, 0), 20
                        )
                        cv2.imshow(f"Almanac AI — Enrolling {student_name}", confirm_frame)
                        cv2.waitKey(300)
                    else:
                        print(f"  Capture rejected: {reason}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if capture_count < self.required_captures:
            return {
                'success': False,
                'captures': capture_count,
                'message': f'Enrollment incomplete. Got {capture_count}/{self.required_captures} captures.'
            }

        # Save embeddings to database
        saved = self._save_embeddings(
            student_id,
            embeddings_collected,
            db
        )

        if saved:
            # Update student enrollment status
            student = db.query(Student).filter(
                Student.id == student_id
            ).first()

            if student:
                student.is_enrolled = True
                db.commit()

            # Log in audit trail
            audit = AuditLog(
                action       = 'STUDENT_ENROLLED',
                target_id    = student_id,
                details      = json.dumps({
                    'student_name': student_name,
                    'captures':     capture_count,
                    'model':        'buffalo_l'
                }),
                performed_by = 'system'
            )
            db.add(audit)
            db.commit()

            print(f"\nEnrollment complete for {student_name}.")
            print(f"{capture_count} face captures stored successfully.")

            return {
                'success': True,
                'captures': capture_count,
                'message': f'Successfully enrolled {student_name} with {capture_count} captures.'
            }

        return {
            'success': False,
            'captures': capture_count,
            'message': 'Failed to save embeddings to database.'
        }

    def _save_embeddings(
        self,
        student_id: str,
        embeddings: list,
        db: Session
    ) -> bool:
        """
        Save all collected embeddings to database.
        Each capture is stored as a separate embedding row.
        Multiple embeddings per student improves accuracy.

        Args:
            student_id:  student UUID
            embeddings:  list of numpy embedding arrays
            db:          database session

        Returns:
            True if saved successfully
        """
        try:
            # Remove old embeddings if re-enrolling
            db.query(FaceEmbedding).filter(
                FaceEmbedding.student_id == student_id
            ).delete()

            for embedding in embeddings:
                face_emb = FaceEmbedding(
                    student_id = student_id,
                    embedding  = self.embedder.to_json(embedding),
                    model_name = 'buffalo_l'
                )
                db.add(face_emb)

            db.commit()
            return True

        except Exception as e:
            print(f"Error saving embeddings: {e}")
            db.rollback()
            return False

    def load_all_embeddings(self, db: Session) -> list:
        """
        Load all enrolled student embeddings from database.
        Used by the recognition engine at startup and
        refreshed periodically during operation.

        Args:
            db: database session

        Returns:
            List of dicts with student_id, student_name, embedding
        """
        results = (
            db.query(FaceEmbedding, Student)
            .join(Student, FaceEmbedding.student_id == Student.id)
            .filter(Student.is_enrolled == True)
            .filter(Student.is_active == True)
            .all()
        )

        enrolled = []
        for face_emb, student in results:
            enrolled.append({
                'student_id':   student.id,
                'student_name': f"{student.first_name} {student.last_name}",
                'class_name':   student.class_name,
                'embedding':    face_emb.embedding
            })

        return enrolled