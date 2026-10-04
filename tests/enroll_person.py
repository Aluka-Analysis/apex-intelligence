"""
Enroll a new person into Almanac AI.
Run this script for each person you want to enroll.
Usage: python tests/enroll_person.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_local_db, LocalSession
from database.models import School, Student
from ai.enrollment.enroller import StudentEnroller


def enroll_new_person():
    """
    Interactive enrollment for a new person.
    Asks for details then opens camera for face capture.
    """
    print("=" * 60)
    print("ALMANAC AI — STUDENT ENROLLMENT")
    print("=" * 60)

    init_local_db()
    db = LocalSession()

    # Get or create school
    school = db.query(School).first()
    if not school:
        school = School(
            name    = 'Greenfield Academy',
            address = 'Port Harcourt, Rivers State',
            phone   = '08012345678'
        )
        db.add(school)
        db.commit()
        print(f"School created: {school.name}")

    print(f"\nSchool: {school.name}")
    print("-" * 60)

    # Collect student details
    print("\nEnter student details:")
    first_name    = input("First name:      ").strip()
    last_name     = input("Last name:       ").strip()
    student_id_no = input("Student ID/No:   ").strip()
    class_name    = input("Class:           ").strip()
    parent_phone  = input("Parent phone:    ").strip()

    if not first_name or not last_name:
        print("First name and last name are required.")
        db.close()
        return

    # Check if already enrolled
    existing = db.query(Student).filter(
        Student.first_name == first_name,
        Student.last_name  == last_name,
        Student.school_id  == school.id
    ).first()

    if existing:
        print(f"\nStudent {first_name} {last_name} already exists.")
        choice = input("Re-enroll with new face captures? (y/n): ").strip().lower()
        if choice != 'y':
            db.close()
            return
        student = existing
    else:
        # Create new student
        student = Student(
            school_id     = school.id,
            student_id_no = student_id_no or f"STU{len(db.query(Student).all())+1:03d}",
            first_name    = first_name,
            last_name     = last_name,
            class_name    = class_name or 'Unknown',
            parent_phone  = parent_phone or ''
        )
        db.add(student)
        db.commit()
        print(f"\nStudent record created: {student.id}")

    print(f"\nReady to enroll: {first_name} {last_name}")
    print("-" * 60)
    print("Instructions:")
    print("  → Position face clearly in camera")
    print("  → Press SPACE to capture")
    print("  → Move slightly between captures")
    print("  → 5 captures required")
    print("  → Try: straight, slight left, slight right")
    print("  → Press Q to cancel")
    input("\nPress ENTER to open camera...")

    enroller = StudentEnroller(required_captures=5)
    result   = enroller.enroll_from_camera(
        student_id   = student.id,
        student_name = f"{first_name} {last_name}",
        db           = db
    )

    print("\n" + "=" * 60)
    print("ENROLLMENT RESULT")
    print("=" * 60)
    print(f"Success:  {result['success']}")
    print(f"Captures: {result['captures']}")
    print(f"Message:  {result['message']}")

    if result['success']:
        print(f"\nStudent {first_name} {last_name} is now enrolled.")
        print("They can be recognised at the school gate.")

    # Show all enrolled students
    print("\n" + "-" * 60)
    print("Currently enrolled students:")
    enroller2  = StudentEnroller()
    all_enrolled = enroller2.load_all_embeddings(db)
    for s in all_enrolled:
        print(f"  → {s['student_name']} — {s['class_name']} ({len(s['embeddings'])} embeddings)")

    db.close()


if __name__ == '__main__':
    enroll_new_person()