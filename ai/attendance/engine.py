"""
Almanac AI Attendance Engine.

Pipeline:

Face Detection
      ↓
Face Quality
      ↓
Face Embedding
      ↓
Identity Matching
      ↓
Passive Liveness
      ↓
Temporal Validation
      ↓
Duplicate Protection
      ↓
Attendance Database

The engine deliberately keeps recognition and liveness
as separate responsibilities.
"""

from datetime import datetime


class AttendanceEngine:

    def __init__(
        self,
        detector,
        embedder,
        matcher,
        liveness_detector,
        real_threshold=0.60,
        min_liveness_frames=5,
        min_live_frames=3
    ):

        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.liveness = liveness_detector

        self.real_threshold = real_threshold
        self.min_liveness_frames = min_liveness_frames
        self.min_live_frames = min_live_frames

        self.logged_students = set()

    # =========================================================
    # PROCESS FRAME
    # =========================================================

    def process_frame(
        self,
        frame,
        enrolled_embeddings,
        db,
        school_id
    ):

        faces = self.detector.detect(frame)

        results = []

        for face in faces:

            # -------------------------------------------------
            # 1. FACE QUALITY
            # -------------------------------------------------

            quality_ok, reason = (
                self.detector.is_face_quality_acceptable(face)
            )

            if not quality_ok:

                results.append({
                    "status": "poor_quality",
                    "reason": reason,
                    "face": face
                })

                continue

            # -------------------------------------------------
            # 2. FACE EMBEDDING
            # -------------------------------------------------

            try:

                embedding = self.embedder.extract(face)

            except Exception as error:

                results.append({
                    "status": "embedding_error",
                    "error": str(error),
                    "face": face
                })

                continue

            # -------------------------------------------------
            # 3. IDENTITY MATCHING
            # -------------------------------------------------

            match = self.matcher.match(
                embedding,
                enrolled_embeddings
            )

            if not match["matched"]:

                results.append({
                    "status": "unknown",
                    "confidence": match["confidence"],
                    "face": face
                })

                continue

            student_id = match["student_id"]
            student_name = match["student_name"]
            recognition_confidence = match["confidence"]

            # -------------------------------------------------
            # 4. PASSIVE LIVENESS
            # -------------------------------------------------

            face_crop = self.liveness.crop_face(
                frame,
                face.bbox
            )

            if face_crop is None:

                results.append({
                    "status": "liveness_error",
                    "student_id": student_id,
                    "student_name": student_name,
                    "confidence": recognition_confidence,
                    "face": face
                })

                continue

            try:

                liveness_result = (
                    self.liveness.observe(
                        face_crop,
                        identity=str(student_id)
                    )
                )

            except Exception as error:

                results.append({
                    "status": "liveness_error",
                    "student_id": student_id,
                    "student_name": student_name,
                    "confidence": recognition_confidence,
                    "error": str(error),
                    "face": face
                })

                continue

            decision = liveness_result["decision"]

            # -------------------------------------------------
            # 5. STILL CHECKING
            # -------------------------------------------------

            if decision == "UNCERTAIN" and \
               liveness_result["status"] == "checking":

                results.append({
                    "status": "checking_liveness",
                    "student_id": student_id,
                    "student_name": student_name,
                    "confidence": recognition_confidence,
                    "real_score":
                        liveness_result["live_score"],
                    "smoothed_score":
                        liveness_result[
                            "smoothed_live_score"
                        ],
                    "frames":
                        liveness_result["frames_seen"],
                    "required_frames":
                        liveness_result["required_frames"],
                    "face": face
                })

                continue

            # -------------------------------------------------
            # 6. SPOOF
            # -------------------------------------------------

            if decision == "SPOOF":

                results.append({
                    "status": "liveness_failed",
                    "reason": "spoof_detected",
                    "student_id": student_id,
                    "student_name": student_name,
                    "confidence": recognition_confidence,
                    "real_score":
                        liveness_result["live_score"],
                    "smoothed_score":
                        liveness_result[
                            "smoothed_live_score"
                        ],
                    "face": face
                })

                continue

            # -------------------------------------------------
            # 7. UNCERTAIN
            # -------------------------------------------------

            if decision == "UNCERTAIN":

                results.append({
                    "status": "liveness_uncertain",
                    "student_id": student_id,
                    "student_name": student_name,
                    "confidence": recognition_confidence,
                    "real_score":
                        liveness_result["live_score"],
                    "smoothed_score":
                        liveness_result[
                            "smoothed_live_score"
                        ],
                    "face": face
                })

                continue

            # -------------------------------------------------
            # 8. LIVE
            # -------------------------------------------------

            if decision != "LIVE":

                continue

            # -------------------------------------------------
            # 9. DUPLICATE PROTECTION
            # -------------------------------------------------

            if student_id in self.logged_students:

                results.append({
                    "status": "already_logged",
                    "student_id": student_id,
                    "student_name": student_name,
                    "confidence": recognition_confidence,
                    "real_score":
                        liveness_result["live_score"],
                    "smoothed_score":
                        liveness_result[
                            "smoothed_live_score"
                        ],
                    "face": face
                })

                continue

            # -------------------------------------------------
            # 10. DATABASE ATTENDANCE
            # -------------------------------------------------

            now = datetime.now()

            try:

                from database.models import AttendanceRecord

                attendance = AttendanceRecord(
                    student_id=student_id,
                    school_id=school_id,
                    date=now.date(),
                    time_recorded=now.time(),
                    status="Present",
                    confidence=recognition_confidence,
                    method="face_recognition_passive_liveness"
                )

                db.add(attendance)
                db.commit()

            except Exception as error:

                db.rollback()

                results.append({
                    "status": "database_error",
                    "student_id": student_id,
                    "student_name": student_name,
                    "error": str(error),
                    "face": face
                })

                continue

            # -------------------------------------------------
            # 11. MARK AS LOGGED
            # -------------------------------------------------

            self.logged_students.add(student_id)

            # Clear liveness history after successful attendance.
            self.liveness.reset_identity(
                str(student_id)
            )

            # -------------------------------------------------
            # 12. SUCCESS
            # -------------------------------------------------

            results.append({
                "status": "attendance_logged",
                "student_id": student_id,
                "student_name": student_name,
                "confidence": recognition_confidence,
                "real_score":
                    liveness_result["live_score"],
                "smoothed_score":
                    liveness_result[
                        "smoothed_live_score"
                    ],
                "timestamp": now,
                "face": face
            })

        return results

    # =========================================================
    # RESET SESSION
    # =========================================================

    def reset_session(self):

        self.logged_students.clear()

        self.liveness.reset_all()