"""
Test live face recognition with persistent liveness detection.
Liveness progress is tied to student identity.
Brief face loss does not reset blink count.
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


def test_live_recognition():
    """
    Live recognition with persistent liveness detection.
    """
    print("=" * 60)
    print("ALMANAC AI — SCHOOL GATE SIMULATION")
    print("Persistent Liveness · Anti-Spoofing · Attendance Logging")
    print("=" * 60)

    init_local_db()
    db = LocalSession()

    enroller = StudentEnroller()
    enrolled = enroller.load_all_embeddings(db)

    if not enrolled:
        print("No enrolled students found.")
        print("Run test_enrollment.py first.")
        db.close()
        return

    print(f"\nPress Q to quit\n")

    # ── Initialize all modules ONCE ──
    detector = FaceDetector()
    matcher  = FaceMatcher(threshold=0.45)
    embedder = FaceEmbedder()
    liveness = LivenessDetector(
        ear_threshold   = 0.25,
        blinks_required = 2,
        session_seconds = 20.0
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access camera.")
        db.close()
        return

    attendance_logged = set()
    frame_count       = 0
    process_every     = 3
    current_match     = None
    display_success   = None
    success_shown_at  = None
    success_duration  = 3.0

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
                    embedding     = embedder.extract(face)
                    match_result  = matcher.match(embedding, enrolled)

                    if match_result['matched']:
                        student_id   = match_result['student_id']
                        student_name = match_result['student_name']
                        class_name   = match_result['class_name']
                        confidence   = match_result['confidence']
                        current_match = match_result

                        # Check if already logged this session
                        if student_id in attendance_logged:
                            display = detector.draw_detection(
                                display, face,
                                label      = f"{student_name} — Already logged",
                                confidence = confidence,
                                color      = (0, 200, 0)
                            )

                        elif liveness.is_student_live(student_id):
                            # Liveness already confirmed — log attendance
                            if student_id not in attendance_logged:
                                attendance_logged.add(student_id)
                                display_success  = match_result
                                success_shown_at = datetime.now()

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

                                print(f"\nATTENDANCE LOGGED:")
                                print(f"  Student:    {student_name}")
                                print(f"  Class:      {class_name}")
                                print(f"  Confidence: {confidence:.1%}")
                                print(f"  Date:       {date_str}")
                                print(f"  Time:       {time_str}")
                                print(f"  Status:     Present\n")

                        else:
                            # Run liveness check
                            live_result = liveness.update(face, student_id)

                            if live_result['is_live']:
                                # Will be logged on next frame
                                pass
                            else:
                                # Still checking liveness
                                color = (0, 165, 255)

                                display = detector.draw_detection(
                                    display, face,
                                    label      = student_name,
                                    confidence = confidence,
                                    color      = color
                                )

                                # Liveness instruction
                                cv2.putText(
                                    display,
                                    live_result['status'],
                                    (10, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, color, 2
                                )

                                # Blink progress bar
                                progress = (
                                    live_result['blink_count'] /
                                    live_result['blinks_required']
                                )
                                bar_w = int(progress * 200)

                                cv2.rectangle(
                                    display,
                                    (10, 120), (210, 138),
                                    (40, 40, 40), -1
                                )
                                if bar_w > 0:
                                    cv2.rectangle(
                                        display,
                                        (10, 120),
                                        (10 + bar_w, 138),
                                        color, -1
                                    )

                                # Time remaining
                                cv2.putText(
                                    display,
                                    f"Time left: {live_result['time_left']:.0f}s",
                                    (10, 158),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (180, 180, 180), 1
                                )

                                # EAR debug info
                                cv2.putText(
                                    display,
                                    f"EAR: {live_result['ear']:.3f}",
                                    (10, 178),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (150, 150, 150), 1
                                )

                    else:
                        # Unknown face
                        if current_match is None:
                            display = detector.draw_detection(
                                display, face,
                                label  = "Unknown",
                                color  = (0, 0, 255)
                            )
                            cv2.putText(
                                display,
                                f"Score: {match_result['confidence']:.1%}",
                                (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 1
                            )

                else:
                    current_match = None

            # ── Success overlay ──
            if display_success and success_shown_at:
                elapsed = (datetime.now() - success_shown_at).total_seconds()

                if elapsed < success_duration:
                    face = detector.detect_largest_face(frame)
                    if face is not None:
                        display = detector.draw_detection(
                            display, face,
                            label = f"{display_success['student_name']} — Present",
                            confidence = display_success['confidence'],
                            color = (0, 255, 0)
                        )
                    cv2.putText(
                        display,
                        "ATTENDANCE RECORDED",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2
                    )
                else:
                    display_success  = None
                    success_shown_at = None

            # ── HUD ──
            overlay = display.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (display.shape[1], 78),
                (0, 0, 0), -1
            )
            cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

            cv2.putText(
                display, date_str,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (220, 220, 220), 1
            )
            cv2.putText(
                display, time_str,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80, (255, 255, 100), 2
            )

            sx = display.shape[1] - 200
            cv2.putText(
                display,
                f"Enrolled: {len(enrolled)}",
                (sx, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (200, 200, 200), 1
            )
            cv2.putText(
                display,
                f"Logged:   {len(attendance_logged)}",
                (sx, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (100, 255, 100), 1
            )

            cv2.putText(
                display, "Q to quit",
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1
            )

            cv2.imshow("Almanac AI — School Gate", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()

    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Date:               {datetime.now().strftime('%A %d %B %Y')}")
    print(f"Enrolled students:  {len(enrolled)}")
    print(f"Students logged:    {len(attendance_logged)}")
    print("=" * 60)


if __name__ == '__main__':
    test_live_recognition()