"""
Almanac AI — School Gate Recognition System
Complete pipeline:
→ Face Detection
→ Identity Matching
→ Single Blink Liveness
→ Attendance Logging
→ Date and Time Display
"""

import sys
import os
import cv2
import sqlalchemy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_local_db, LocalSession
from database.models import AttendanceRecord
from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder
from ai.recognition.matcher import FaceMatcher
from ai.recognition.liveness import LivenessDetector
from ai.enrollment.enroller import StudentEnroller
from datetime import datetime


def get_datetime_display() -> tuple:
    now = datetime.now()
    return (
        now.strftime('%A, %d %B %Y'),
        now.strftime('%H:%M:%S')
    )


def draw_hud(display, date_str, time_str, enrolled_count, logged_count):
    """Draw heads-up display on camera frame."""
    h, w = display.shape[:2]

    # Dark header strip
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # Date
    cv2.putText(
        display, date_str,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (220, 220, 220), 1
    )

    # Time
    cv2.putText(
        display, time_str,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80, (255, 255, 100), 2
    )

    # Stats top right
    sx = w - 200
    cv2.putText(
        display,
        f"Enrolled: {enrolled_count}",
        (sx, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (200, 200, 200), 1
    )
    cv2.putText(
        display,
        f"Logged:   {logged_count}",
        (sx, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (100, 255, 100), 1
    )

    # Bottom instruction
    cv2.putText(
        display, "Q to quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (180, 180, 180), 1
    )

    return display


def draw_liveness_ui(display, face, student_name, confidence, live_result, detector):
    """Draw liveness check UI on frame."""
    color = (0, 165, 255)

    display = detector.draw_detection(
        display, face,
        label      = student_name,
        confidence = confidence,
        color      = color
    )

    # Instruction
    cv2.putText(
        display,
        live_result['status'],
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70, color, 2
    )

    # Progress bar
    progress = min(
        live_result['blink_count'] /
        live_result['blinks_required'],
        1.0
    )
    bar_w = int(progress * 220)

    cv2.rectangle(display, (10, 118), (230, 136), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(display, (10, 118), (10 + bar_w, 136), color, -1)

    # Time remaining
    cv2.putText(
        display,
        f"Time: {live_result['time_left']:.0f}s",
        (10, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50, (180, 180, 180), 1
    )

    # EAR value for debugging
    cv2.putText(
        display,
        f"EAR: {live_result['ear']:.3f}",
        (10, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (140, 140, 140), 1
    )

    return display


def draw_success_ui(display, face, student_name, confidence, detector):
    """Draw success confirmation UI."""
    display = detector.draw_detection(
        display, face,
        label      = f"{student_name} — Present",
        confidence = confidence,
        color      = (0, 255, 0)
    )
    cv2.putText(
        display,
        "ATTENDANCE RECORDED",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 255, 0), 2
    )
    return display


def log_attendance(db, student_id, confidence, date_str, time_str, class_name, student_name):
    """Write attendance record to database."""
    school_id = db.execute(
        sqlalchemy.text(
            "SELECT school_id FROM students WHERE id = :sid"
        ),
        {'sid': student_id}
    ).scalar()

    record = AttendanceRecord(
        student_id = student_id,
        school_id  = school_id,
        date       = datetime.now().strftime('%Y-%m-%d'),
        status     = 'present',
        confidence = confidence,
        method     = 'face_recognition'
    )
    db.add(record)
    db.commit()

    print("=" * 50)
    print("ATTENDANCE LOGGED")
    print("=" * 50)
    print(f"  Student:    {student_name}")
    print(f"  Class:      {class_name}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Date:       {date_str}")
    print(f"  Time:       {time_str}")
    print(f"  Liveness:   Confirmed (1 blink)")
    print(f"  Status:     Present")
    print("=" * 50 + "\n")


def run_gate():
    """
    Main school gate recognition loop.

    States:
    DETECTING      → scanning for known faces
    LIVENESS_CHECK → waiting for blink confirmation
    SUCCESS        → showing confirmed result
    ALREADY_LOGGED → student already marked present
    """
    print("=" * 60)
    print("ALMANAC AI — SCHOOL GATE SYSTEM")
    print("Identity Matching + Liveness Detection")
    print("=" * 60)

    init_local_db()
    db = LocalSession()

    # Load enrolled students
    enroller = StudentEnroller()
    enrolled = enroller.load_all_embeddings(db)

    if not enrolled:
        print("No enrolled students found.")
        print("Run tests/test_enrollment.py first.")
        db.close()
        return

    print(f"\nPress Q to quit\n")

    # Initialize all modules once
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher  = FaceMatcher(threshold=0.45)
    liveness = LivenessDetector(
        ear_threshold   = 0.25,
        blinks_required = 1,
        session_seconds = 8.0
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        db.close()
        return

    # State
    attendance_logged = set()
    frame_count       = 0
    process_every     = 3
    current_match     = None
    state             = 'DETECTING'
    success_shown_at  = None
    success_duration  = 2.5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.flip(frame, 1)
            display      = frame.copy()
            frame_count += 1
            date_str, time_str = get_datetime_display()

            if frame_count % process_every == 0:
                face = detector.detect_largest_face(frame)

                if face is not None:
                    embedding    = embedder.extract(face)
                    match_result = matcher.match(embedding, enrolled)

                    if not match_result['matched']:
                        # Unknown face
                        state         = 'DETECTING'
                        current_match = None
                        display = detector.draw_detection(
                            display, face,
                            label = "Unknown",
                            color = (0, 0, 255)
                        )

                    else:
                        student_id   = match_result['student_id']
                        student_name = match_result['student_name']
                        class_name   = match_result['class_name']
                        confidence   = match_result['confidence']
                        current_match = match_result

                        if student_id in attendance_logged:
                            # Already logged this session
                            state = 'ALREADY_LOGGED'
                            display = detector.draw_detection(
                                display, face,
                                label      = f"{student_name} — Already logged",
                                confidence = confidence,
                                color      = (0, 200, 100)
                            )

                        elif liveness.is_student_live(student_id):
                            # Liveness confirmed — log and show success
                            if student_id not in attendance_logged:
                                attendance_logged.add(student_id)
                                log_attendance(
                                    db, student_id, confidence,
                                    date_str, time_str,
                                    class_name, student_name
                                )
                                state            = 'SUCCESS'
                                success_shown_at = datetime.now()

                        else:
                            # Run liveness check
                            state       = 'LIVENESS_CHECK'
                            live_result = liveness.update(face, student_id)
                            display     = draw_liveness_ui(
                                display, face, student_name,
                                confidence, live_result, detector
                            )

                else:
                    # No face detected
                    if state == 'LIVENESS_CHECK':
                        pass  # Keep liveness session alive briefly
                    else:
                        state         = 'DETECTING'
                        current_match = None

            # Success overlay
            if state == 'SUCCESS' and success_shown_at and current_match:
                elapsed = (datetime.now() - success_shown_at).total_seconds()
                if elapsed < success_duration:
                    face = detector.detect_largest_face(frame)
                    if face is not None:
                        display = draw_success_ui(
                            display, face,
                            current_match['student_name'],
                            current_match['confidence'],
                            detector
                        )
                else:
                    state            = 'DETECTING'
                    current_match    = None
                    success_shown_at = None

            # State indicator bottom left
            state_colors = {
                'DETECTING':      (150, 150, 150),
                'LIVENESS_CHECK': (0, 165, 255),
                'SUCCESS':        (0, 255, 0),
                'ALREADY_LOGGED': (0, 200, 100),
            }
            cv2.putText(
                display,
                f"State: {state}",
                (display.shape[1] - 220, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, state_colors.get(state, (150, 150, 150)), 1
            )

            # Draw HUD
            display = draw_hud(
                display, date_str, time_str,
                len(enrolled), len(attendance_logged)
            )

            cv2.imshow("Almanac AI — School Gate", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Date:              {datetime.now().strftime('%A %d %B %Y')}")
    print(f"Enrolled students: {len(enrolled)}")
    print(f"Students logged:   {len(attendance_logged)}")
    print("=" * 60)


if __name__ == '__main__':
    run_gate()