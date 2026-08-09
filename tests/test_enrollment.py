"""
Test live camera enrollment.
Run this to enroll yourself as a test student
and verify the complete enrollment pipeline works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_local_db, LocalSession
from database.models import Student
from ai.enrollment.enroller import StudentEnroller


def test_live_enrollment():
    """
    Enroll a test student using the laptop camera.
    """
    print("=" * 55)
    print("ALMANAC AI — LIVE ENROLLMENT TEST")
    print("=" * 55)

    # Initialize database
    init_local_db()
    db = LocalSession()

    # Find Chidi Okafor from our earlier test
    student = db.query(Student).filter(
        Student.first_name == 'Chidi',
        Student.last_name  == 'Okafor'
    ).first()

    if not student:
        print("Test student not found.")
        print("Run the setup command first.")
        db.close()
        return

    print(f"\nEnrolling: {student.first_name} {student.last_name}")
    print(f"Class:     {student.class_name}")
    print(f"ID:        {student.id}")
    print("\nInstructions:")
    print("→ Position your face clearly in the camera")
    print("→ Press SPACE to capture each image")
    print("→ Move slightly between captures")
    print("→ Try different angles: straight, left, right")
    print("→ 5 captures required")
    print("→ Press Q to cancel\n")

    enroller = StudentEnroller(required_captures=5)

    result = enroller.enroll_from_camera(
        student_id   = student.id,
        student_name = f"{student.first_name} {student.last_name}",
        db           = db
    )

    print("\n" + "=" * 55)
    print("ENROLLMENT RESULT")
    print("=" * 55)
    print(f"Success:  {result['success']}")
    print(f"Captures: {result['captures']}")
    print(f"Message:  {result['message']}")

    if result['success']:
        # Verify in database
        updated = db.query(Student).filter(
            Student.id == student.id
        ).first()
        print(f"\nDatabase confirmed:")
        print(f"→ Enrolled: {updated.is_enrolled}")

        # Load embeddings to verify storage
        embedder_count = db.execute(
            __import__('sqlalchemy').text(
                "SELECT COUNT(*) FROM face_embeddings WHERE student_id = :sid"
            ),
            {'sid': student.id}
        ).scalar()
        print(f"→ Embeddings stored: {embedder_count}")

    db.close()
    print("=" * 55)


if __name__ == '__main__':
    test_live_enrollment()