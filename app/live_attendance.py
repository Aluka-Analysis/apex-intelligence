"""
ALMANAC AI — LIVE ATTENDANCE

Production-oriented Proof of Concept.

Hardware:
    Laptop webcam

Recognition:
    InsightFace buffalo_l

Liveness:
    MiniFASNetV2

Storage:
    Local SQLite

Current Mode:
    DIAGNOSTIC LIVENESS MODE

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
    Diagnostic Output

IMPORTANT:

    Attendance logging is currently DISABLED.

    The current objective is to validate the MiniFASNetV2
    class mapping and liveness behaviour using a real
    person in front of the laptop webcam.

    Do NOT use the current liveness score as a production
    security decision until the MiniFASNetV2 class mapping
    has been experimentally validated.
"""

import cv2
import time

from collections import defaultdict

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

# Face recognition threshold.
#
# Your current system was already recognizing Chidi,
# so we leave this at the existing value.
RECOGNITION_THRESHOLD = 0.45

# ------------------------------------------------------------
# LIVENESS CONFIGURATION
# ------------------------------------------------------------
#
# IMPORTANT:
#
# At this stage the liveness engine is diagnostic.
#
# We are NOT using this threshold to decide attendance.
# It is only used by AttendanceEngine to display whether
# the currently assumed "real" class is above or below
# the diagnostic threshold.
#
# The current engine assumes class 1 temporarily.
# Your logs show class 2 is actually dominant.
#
# We therefore keep the threshold here but DO NOT claim
# that it represents a validated live-person probability.
#
LIVENESS_REAL_THRESHOLD = 0.60

MIN_LIVENESS_FRAMES = 3

# Number of frames that should be considered live before
# the diagnostic engine reports an above-threshold result.
MIN_LIVE_FRAMES = 3

WINDOW_NAME = (
    "Almanac AI — Live Attendance"
)


# ============================================================
# LOAD ENROLLED STUDENTS
# ============================================================

def load_enrolled_embeddings(db):
    """
    Load all active enrolled students and their face
    embeddings from the local SQLite database.

    Returns:

        [
            {
                "student_id": ...,
                "student_name": ...,
                "class_name": ...,
                "embeddings": [...]
            }
        ]
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
# DRAW RESULT
# ============================================================

def draw_result(
    frame,
    result
):
    """
    Draw recognition and diagnostic liveness information
    onto the webcam frame.
    """

    face = result.get(
        "face"
    )

    if face is None:
        return frame

    # --------------------------------------------------------
    # FACE BOUNDING BOX
    # --------------------------------------------------------

    try:

        x1, y1, x2, y2 = [
            int(value)
            for value in face.bbox
        ]

    except Exception:

        return frame

    # --------------------------------------------------------
    # RESULT DATA
    # --------------------------------------------------------

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

    class_0 = result.get(
        "class_0"
    )

    class_1 = result.get(
        "class_1"
    )

    class_2 = result.get(
        "class_2"
    )

    frames = result.get(
        "frames"
    )

    required_frames = result.get(
        "required_frames"
    )

    verdict = result.get(
        "verdict"
    )

    # ========================================================
    # STATUS LABEL
    # ========================================================

    if status == "liveness_diagnostic":

        if verdict == "CURRENTLY_ABOVE_THRESHOLD":

            label = (
                f"{student_name} | "
                "LIVENESS ABOVE THRESHOLD"
            )

        elif verdict == "CURRENTLY_BELOW_THRESHOLD":

            label = (
                f"{student_name} | "
                "LIVENESS BELOW THRESHOLD"
            )

        else:

            label = (
                f"{student_name} | "
                "CHECKING LIVENESS..."
            )

    elif status == "unknown":

        label = "Unknown"

    elif status == "poor_quality":

        label = "Poor face quality"

    elif status == "embedding_error":

        label = "Embedding error"

    elif status == "recognition_error":

        label = "Recognition error"

    elif status == "liveness_error":

        label = (
            f"{student_name} | "
            "Liveness error"
        )

    else:

        label = status

    # ========================================================
    # BOX COLOR
    # ========================================================

    if (
        status == "liveness_diagnostic"
        and verdict == "CURRENTLY_ABOVE_THRESHOLD"
    ):

        box_color = (
            0,
            255,
            0
        )

    elif (
        status == "liveness_diagnostic"
        and verdict == "CURRENTLY_BELOW_THRESHOLD"
    ):

        box_color = (
            0,
            0,
            255
        )

    elif status == "liveness_diagnostic":

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

    elif status == "poor_quality":

        box_color = (
            0,
            165,
            255
        )

    elif status == "liveness_error":

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

    # ========================================================
    # FACE BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    # ========================================================
    # STATUS LABEL
    # ========================================================

    cv2.putText(
        frame,
        label,
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

    # ========================================================
    # RECOGNITION CONFIDENCE
    # ========================================================

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

    # ========================================================
    # LIVENESS SCORES
    # ========================================================

    if real_score is not None:

        cv2.putText(
            frame,
            (
                f"Current real score: "
                f"{real_score:.2%}"
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
                f"{smoothed_score:.2%}"
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

    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================
    #
    # This is especially important right now because we are
    # trying to determine which MiniFASNetV2 class represents
    # a real/live face.
    #

    if class_0 is not None:

        cv2.putText(
            frame,
            (
                f"C0: "
                f"{class_0:.2%}"
            ),
            (
                x1,
                y2 + 80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1
        )

    if class_1 is not None:

        cv2.putText(
            frame,
            (
                f"C1: "
                f"{class_1:.2%}"
            ),
            (
                x1 + 90,
                y2 + 80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1
        )

    if class_2 is not None:

        cv2.putText(
            frame,
            (
                f"C2: "
                f"{class_2:.2%}"
            ),
            (
                x1 + 180,
                y2 + 80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1
        )

    # ========================================================
    # TEMPORAL INFORMATION
    # ========================================================

    if (
        frames is not None
        and required_frames is not None
    ):

        cv2.putText(
            frame,
            (
                f"Frames: "
                f"{frames}/"
                f"{required_frames}"
            ),
            (
                x1,
                y2 + 100
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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
        "Recognition: InsightFace buffalo_l"
    )

    print(
        "Liveness: MiniFASNetV2"
    )

    print(
        "Storage: Local SQLite"
    )

    print(
        "Mode: DIAGNOSTIC LIVENESS"
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
        f"\nSchool ID: "
        f"{school_id}"
    )

    # ========================================================
    # LOAD ENROLLED STUDENTS
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
        min_liveness_frames=MIN_LIVENESS_FRAMES,
        min_live_frames=MIN_LIVE_FRAMES
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

    # --------------------------------------------------------
    # CAMERA SETTINGS
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

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
        "\nIMPORTANT:"
    )

    print(
        "Attendance logging is currently DISABLED."
    )

    print(
        "The current session is collecting "
        "MiniFASNetV2 diagnostic data."
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

            # ------------------------------------------------
            # MIRROR CAMERA
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            # ------------------------------------------------
            # PROCESS FRAME
            # ------------------------------------------------

            try:

                results = (
                    engine.process_frame(
                        frame,
                        enrolled_students,
                        db,
                        school_id
                    )
                )

            except Exception as error:

                print()
                print(
                    "[FRAME PROCESSING ERROR]"
                )

                print(
                    str(error)
                )

                results = []

            # ------------------------------------------------
            # DRAW RESULTS
            # ------------------------------------------------

            for result in results:

                status = result.get(
                    "status"
                )

                # ------------------------------------------------
                # RECOGNITION TRACKING
                # ------------------------------------------------

                if status in (
                    "matched",
                    "checking_liveness",
                    "liveness_diagnostic",
                    "liveness_error",
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

                # ------------------------------------------------
                # ATTENDANCE
                # ------------------------------------------------
                #
                # Intentionally disabled.
                #
                # The current AttendanceEngine is in diagnostic
                # mode and does not create attendance records.
                #

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

                # ------------------------------------------------
                # DRAW
                # ------------------------------------------------

                frame = draw_result(
                    frame,
                    result
                )

            # ====================================================
            # SYSTEM STATUS
            # ====================================================

            cv2.putText(
                frame,
                (
                    "ALMANAC AI | "
                    "DIAGNOSTIC LIVENESS"
                ),
                (
                    10,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            # ------------------------------------------------
            # RECOGNIZED STUDENTS
            # ------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"Recognized: "
                    f"{len(students_recognised)}"
                ),
                (
                    10,
                    55
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                1
            )

            # ------------------------------------------------
            # ATTENDANCE STATUS
            # ------------------------------------------------

            cv2.putText(
                frame,
                (
                    "Attendance logging: DISABLED"
                ),
                (
                    10,
                    80
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    0,
                    200,
                    255
                ),
                1
            )

            # ------------------------------------------------
            # QUIT INSTRUCTION
            # ------------------------------------------------

            cv2.putText(
                frame,
                "Q = Quit",
                (
                    10,
                    frame.shape[0] - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (
                    200,
                    200,
                    200
                ),
                1
            )

            # ====================================================
            # DISPLAY
            # ====================================================

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

        # ====================================================
        # CLEANUP
        # ====================================================

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