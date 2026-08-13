"""
ALMANAC AI — LIVE ATTENDANCE

Production-oriented Proof of Concept.

Hardware:
    Laptop webcam

Recognition:
    InsightFace buffalo_l

Liveness:
    Passive MiniFASNetV2

Storage:
    Local SQLite

Pipeline:

    Camera
       ↓
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
    Temporal Smoothing
       ↓
    Duplicate Protection
       ↓
    Attendance Record
       ↓
    SQLite
"""

import cv2
import time

from collections import defaultdict
from datetime import datetime

from database.connection import (
    init_local_db,
    LocalSession
)

from database.models import (
    Student,
    FaceEmbedding,
    School
)

from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder
from ai.recognition.matcher import FaceMatcher

from ai.liveness.passive_liveness import (
    PassiveLivenessDetector
)

from ai.attendance.engine import AttendanceEngine


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

RECOGNITION_THRESHOLD = 0.45

LIVENESS_REAL_THRESHOLD = 0.60

MIN_LIVENESS_FRAMES = 3

WINDOW_NAME = "Almanac AI — Live Attendance"


# ============================================================
# LOAD ENROLLED STUDENTS
# ============================================================

def load_enrolled_embeddings(db):
    """
    Load all active enrolled students.

    IMPORTANT:

    A student can have multiple face embeddings.

    Example:

        Chidi Okafor
            ├── embedding 1
            ├── embedding 2
            ├── embedding 3
            ├── embedding 4
            └── embedding 5

    The embeddings are grouped by student before
    being passed to FaceMatcher.
    """

    rows = (
        db.query(
            FaceEmbedding,
            Student
        )
        .join(
            Student,
            FaceEmbedding.student_id
            == Student.id
        )
        .filter(
            Student.is_enrolled == True
        )
        .filter(
            Student.is_active == True
        )
        .all()
    )

    students = defaultdict(
        lambda: {
            "student_id": None,
            "student_name": None,
            "class_name": None,
            "embeddings": []
        }
    )

    for face_embedding, student in rows:

        student_key = str(
            student.id
        )

        students[student_key][
            "student_id"
        ] = student.id

        students[student_key][
            "student_name"
        ] = (
            f"{student.first_name} "
            f"{student.last_name}"
        )

        students[student_key][
            "class_name"
        ] = student.class_name

        students[student_key][
            "embeddings"
        ].append(
            face_embedding.embedding
        )

    return list(
        students.values()
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

def draw_result(
    frame,
    result
):
    """
    Draw recognition/liveness result
    on the camera frame.
    """

    face = result.get(
        "face"
    )

    if face is None:
        return frame

    x1, y1, x2, y2 = [
        int(value)
        for value in face.bbox
    ]

    status = result.get(
        "status",
        "unknown"
    )

    student_name = result.get(
        "student_name",
        ""
    )

    confidence = result.get(
        "confidence",
        0.0
    )

    real_score = result.get(
        "real_score"
    )

    smoothed_score = result.get(
        "smoothed_score"
    )

    # --------------------------------------------------------
    # Status text
    # --------------------------------------------------------

    if status == "attendance_logged":

        label = (
            f"{student_name} | "
            f"ATTENDANCE LOGGED"
        )

    elif status == "already_logged":

        label = (
            f"{student_name} | "
            f"ALREADY PRESENT"
        )

    elif status == "checking_liveness":

        label = (
            f"{student_name} | "
            f"Checking liveness..."
        )

    elif status == "liveness_failed":

        label = (
            f"{student_name} | "
            f"Liveness failed"
        )

    elif status == "unknown":

        label = "Unknown"

    elif status == "poor_quality":

        label = (
            "Poor face quality"
        )

    else:

        label = status

    # --------------------------------------------------------
    # Draw box
    # --------------------------------------------------------

    if status == "attendance_logged":

        box_color = (
            0,
            255,
            0
        )

    elif status == "liveness_failed":

        box_color = (
            0,
            0,
            255
        )

    elif status == "checking_liveness":

        box_color = (
            0,
            165,
            255
        )

    elif status == "unknown":

        box_color = (
            0,
            0,
            255
        )

    else:

        box_color = (
            255,
            255,
            0
        )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    # --------------------------------------------------------
    # Recognition confidence
    # --------------------------------------------------------

    label_1 = (
        f"{label}"
    )

    cv2.putText(
        frame,
        label_1,
        (
            x1,
            max(
                25,
                y1 - 30
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2
    )

    # --------------------------------------------------------
    # Recognition score
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            f"Recognition: "
            f"{confidence:.1%}"
        ),
        (
            x1,
            y2 + 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        box_color,
        1
    )

    # --------------------------------------------------------
    # Liveness score
    # --------------------------------------------------------

    if real_score is not None:

        cv2.putText(
            frame,
            (
                f"Live: "
                f"{real_score:.1%}"
            ),
            (
                x1,
                y2 + 40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1
        )

    if smoothed_score is not None:

        cv2.putText(
            frame,
            (
                f"Smoothed: "
                f"{smoothed_score:.1%}"
            ),
            (
                x1,
                y2 + 60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 65
    )
    print(
        "ALMANAC AI — LIVE ATTENDANCE"
    )
    print(
        "=" * 65
    )

    print(
        "Production-oriented Proof of Concept"
    )

    print(
        "Hardware: Laptop webcam"
    )

    print(
        "Liveness: Passive"
    )

    print(
        "Storage: Local SQLite"
    )

    print(
        "=" * 65
    )

    # ========================================================
    # DATABASE
    # ========================================================

    print(
        "\nInitializing local database..."
    )

    init_local_db()

    db = LocalSession()

    # ========================================================
    # SCHOOL
    # ========================================================

    school = (
        db.query(School)
        .first()
    )

    if school is None:

        print(
            "ERROR: No school found."
        )

        db.close()

        return

    school_id = school.id

    print(
        f"\nSchool ID: {school_id}"
    )

    # ========================================================
    # LOAD ENROLLMENTS
    # ========================================================

    print(
        "\nLoading enrolled face embeddings..."
    )

    enrolled_students = (
        load_enrolled_embeddings(
            db
        )
    )

    total_embeddings = sum(
        len(
            student["embeddings"]
        )
        for student
        in enrolled_students
    )

    print(
        f"Loaded "
        f"{len(enrolled_students)} "
        f"enrolled students."
    )

    print(
        f"Loaded "
        f"{total_embeddings} "
        f"face embeddings."
    )

    for student in enrolled_students:

        print(
            f"  → "
            f"{student['student_name']} "
            f"({len(student['embeddings'])} "
            f"embeddings)"
        )

    if not enrolled_students:

        print(
            "\nNo enrolled students found."
        )

        db.close()

        return

    # ========================================================
    # FACE DETECTOR
    # ========================================================

    print(
        "\nInitializing face detector..."
    )

    detector = FaceDetector()

    # ========================================================
    # FACE EMBEDDER
    # ========================================================

    print(
        "Initializing face embedder..."
    )

    embedder = FaceEmbedder()

    # ========================================================
    # FACE MATCHER
    # ========================================================

    print(
        "Initializing face matcher..."
    )

    matcher = FaceMatcher(
        threshold=RECOGNITION_THRESHOLD
    )

    # ========================================================
    # PASSIVE LIVENESS
    # ========================================================

    print(
        "Initializing passive liveness..."
    )

    liveness_detector = (
        PassiveLivenessDetector()
    )

    # ========================================================
    # ATTENDANCE ENGINE
    # ========================================================

    print(
        "Initializing attendance engine..."
    )

    engine = AttendanceEngine(
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        liveness_detector=liveness_detector,
        real_threshold=LIVENESS_REAL_THRESHOLD,
        min_liveness_frames=MIN_LIVENESS_FRAMES
    )

    print(
        "\nAll AI components initialized successfully."
    )

    # ========================================================
    # CAMERA
    # ========================================================

    print(
        "\nOpening laptop webcam..."
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        db.close()

        return

    print(
        "\nCamera started successfully."
    )

    print(
        "Look toward the camera normally."
    )

    print(
        "No blinking required."
    )

    print(
        "No head movement required."
    )

    print(
        "Passive liveness is running silently."
    )

    print(
        "\nPress Q to quit."
    )

    print()

    # ========================================================
    # SESSION METRICS
    # ========================================================

    students_recognised = set()

    attendance_saved = 0

    session_start = time.time()

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    try:

        while True:

            ret, frame = camera.read()

            if not ret:

                print(
                    "Failed to read camera frame."
                )

                break

            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            # ------------------------------------------------
            # PROCESS FRAME
            # ------------------------------------------------

            results = (
                engine.process_frame(
                    frame,
                    enrolled_students,
                    db,
                    school_id
                )
            )

            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            for result in results:

                status = result.get(
                    "status"
                )

                if status in (
                    "matched",
                    "checking_liveness",
                    "attendance_logged",
                    "already_logged",
                    "liveness_failed"
                ):

                    student_id = (
                        result.get(
                            "student_id"
                        )
                    )

                    if student_id:

                        students_recognised.add(
                            student_id
                        )

                if status == (
                    "attendance_logged"
                ):

                    attendance_saved += 1

                    student_name = (
                        result.get(
                            "student_name",
                            "Unknown"
                        )
                    )

                    confidence = (
                        result.get(
                            "confidence",
                            0.0
                        )
                    )

                    real_score = (
                        result.get(
                            "real_score",
                            0.0
                        )
                    )

                    timestamp = (
                        result.get(
                            "timestamp"
                        )
                    )

                    print()
                    print(
                        "=" * 60
                    )

                    print(
                        "ATTENDANCE LOGGED"
                    )

                    print(
                        f"Student: "
                        f"{student_name}"
                    )

                    print(
                        f"Confidence: "
                        f"{confidence:.1%}"
                    )

                    print(
                        f"Live score: "
                        f"{real_score:.1%}"
                    )

                    print(
                        f"Time: "
                        f"{timestamp}"
                    )

                    print(
                        "Liveness: Confirmed"
                    )

                    print(
                        "Status: Present"
                    )

                    print(
                        "=" * 60
                    )

                frame = draw_result(
                    frame,
                    result
                )

            # ------------------------------------------------
            # SYSTEM STATUS
            # ------------------------------------------------

            cv2.putText(
                frame,
                (
                    "ALMANAC AI | "
                    "PASSIVE LIVENESS"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    "Q = Quit"
                ),
                (
                    10,
                    frame.shape[0] - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            # ------------------------------------------------
            # SHOW FRAME
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        # ====================================================
        # SESSION SUMMARY
        # ====================================================

        elapsed = (
            time.time()
            - session_start
        )

        print()
        print(
            "=" * 65
        )

        print(
            "ALMANAC AI — SESSION SUMMARY"
        )

        print(
            "=" * 65
        )

        print(
            f"Enrolled students: "
            f"{len(enrolled_students)}"
        )

        print(
            f"Face embeddings: "
            f"{total_embeddings}"
        )

        print(
            f"Students recognised: "
            f"{len(students_recognised)}"
        )

        print(
            f"Attendance records saved: "
            f"{attendance_saved}"
        )

        print(
            f"Session duration: "
            f"{elapsed:.1f} seconds"
        )

        print(
            "=" * 65
        )

        db.close()

        print(
            "\nAttendance session ended."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()