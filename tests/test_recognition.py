"""
Test live face recognition against enrolled students.
This is the school gate simulation.
Almanac AI sees a face and identifies who it is.
"""

import sys
import os
import cv2
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_local_db, LocalSession
from database.models import AttendanceRecord
from ai.recognition.detector import FaceDetector
from ai.recognition.matcher import FaceMatcher
from ai.enrollment.enroller import StudentEnroller
from datetime import datetime


def test_live_recognition():
    """
    Live camera recognition against enrolled students.
    Simulates the school gate experience.
    """
    print("=" * 55)
    print("ALMANAC AI — LIVE RECOGNITION TEST")
    print("Simulating school gate experience")
    print("=" * 55)

    init_local_db()
    db = LocalSession()

    # Load all enrolled embeddings
    enroller = StudentEnroller()
    enrolled = enroller.load_all_embeddings(db)

    if not enrolled:
        print("No enrolled students found.")
        print("Run test_enrollment.py first.")
        db.close()
        return

    print(f"\nLoaded {len(enrolled)} enrolled student(s)")
    for e in enrolled:
        print(f"  → {e['student_name']} — {e['class_name']}")

    print("\nStarting camera — Press Q to quit\n")

    detector  = FaceDetector()
    matcher   = FaceMatcher(threshold=0.45)
    cap       = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access camera.")
        db.close()
        return

    attendance_logged = set()
    frame_count       = 0
    process_every     = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.flip(frame, 1)
            display      = frame.copy()
            frame_count += 1

            # Process every N frames for performance
            if frame_count % process_every == 0:

                face = detector.detect_largest_face(frame)

                if face is not None:
                    from ai.recognition.embedder import FaceEmbedder
                    embedder  = FaceEmbedder()
                    embedding = embedder.extract(face)
                    result    = matcher.match(embedding, enrolled)

                    if result['matched']:
                        name       = result['student_name']
                        confidence = result['confidence']
                        student_id = result['student_id']

                        # Draw green box — known student
                        display = detector.draw_detection(
                            display, face,
                            label      = name,
                            confidence = confidence,
                            color      = (0, 255, 0)
                        )

                        # Log attendance once per session
                        if student_id not in attendance_logged:
                            attendance_logged.add(student_id)

                            record = AttendanceRecord(
                                student_id  = student_id,
                                school_id   = db.execute(
                                    __import__('sqlalchemy').text(
                                        "SELECT school_id FROM students WHERE id = :sid"
                                    ),
                                    {'sid': student_id}
                                ).scalar(),
                                date        = datetime.now().strftime('%Y-%m-%d'),
                                status      = 'present',
                                confidence  = confidence,
                                method      = 'face_recognition'
                            )
                            db.add(record)
                            db.commit()

                            print(f"ATTENDANCE LOGGED:")
                            print(f"  Student:    {name}")
                            print(f"  Confidence: {confidence:.1%}")
                            print(f"  Time:       {datetime.now().strftime('%H:%M:%S')}")
                            print(f"  Status:     Present\n")

                    else:
                        confidence = result['confidence']

                        # Draw red box — unknown face
                        display = detector.draw_detection(
                            display, face,
                            label  = "Unknown",
                            color  = (0, 0, 255)
                        )

                        cv2.putText(
                            display,
                            f"Score: {confidence:.1%}",
                            (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1
                        )

            # Status display
            cv2.putText(
                display,
                f"Students enrolled: {len(enrolled)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )
            cv2.putText(
                display,
                f"Attendance logged: {len(attendance_logged)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )
            cv2.putText(
                display,
                "Q to quit",
                (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )

            cv2.imshow("Almanac AI — School Gate Simulation", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()

    print("=" * 55)
    print("SESSION SUMMARY")
    print("=" * 55)
    print(f"Students recognised: {len(attendance_logged)}")
    print(f"Attendance records created: {len(attendance_logged)}")
    print("=" * 55)


if __name__ == '__main__':
    test_live_recognition()