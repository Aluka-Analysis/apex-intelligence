#!/usr/bin/env python
"""
Database testing script for Apex Intelligence.
Run this to verify your database setup.
"""

def test_database():
    print("=" * 60)
    print("🧪 TESTING DATABASE SETUP")
    print("=" * 60)
    
    # Test 1: Initialize database
    print("\n📌 Test 1: Initialize Database")
    from database.connection import init_local_db
    init_local_db()
    print("✅ Database initialized")
    
    # Test 2: Check tables
    print("\n📌 Test 2: Check Tables")
    import sqlite3
    conn = sqlite3.connect('data/apex_local.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"✅ Found {len(tables)} tables:")
    for table in tables:
        print(f"   - {table[0]}")
    conn.close()
    
    # Test 3: Insert test data
    print("\n📌 Test 3: Insert Test Data")
    from database.connection import LocalSession
    from database.models import School, Student
    import uuid
    
    db = LocalSession()
    school = School(
        id=str(uuid.uuid4()),
        name='Test School',
        address='123 Test St'
    )
    db.add(school)
    db.commit()
    print(f"✅ Created test school: {school.id}")
    
    student = Student(
        id=str(uuid.uuid4()),
        school_id=school.id,
        student_id_no='TEST001',
        first_name='John',
        last_name='Doe',
        is_enrolled=True
    )
    db.add(student)
    db.commit()
    print(f"✅ Created test student: {student.id}")
    db.close()
    
    # Test 4: Test audit
    print("\n📌 Test 4: Test Audit Log")
    from database.audit import log_action
    log_action('test_audit', 'tester@apex.com', 'TEST123', {'test': True})
    print("✅ Audit log created")
    
    # Test 5: Query data
    print("\n📌 Test 5: Query Data")
    db = LocalSession()
    students = db.query(Student).all()
    print(f"✅ Found {len(students)} students in database")
    for s in students:
        print(f"   - {s.first_name} {s.last_name} ({s.student_id_no})")
    db.close()
    
    # Test 6: Clean up
    print("\n📌 Test 6: Clean Up Test Data")
    db = LocalSession()
    from database.models import AuditLog
    db.query(Student).filter(Student.student_id_no == 'TEST001').delete()
    db.query(School).filter(School.name == 'Test School').delete()
    db.query(AuditLog).filter(AuditLog.action == 'test_audit').delete()
    db.commit()
    db.close()
    print("✅ Test data cleaned up")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Database is ready!")
    print("=" * 60)

if __name__ == "__main__":
    test_database()