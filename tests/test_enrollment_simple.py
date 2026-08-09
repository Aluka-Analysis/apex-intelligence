"""
Simple Live Enrollment Test with Better Camera Handling.
This version is optimized for Windows.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from database.connection import init_local_db, LocalSession
from database.models import Student
from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder


def test_enrollment_simple():
    """
    Simple enrollment with optimized camera handling.
    """
    print("=" * 55)
    print("ALMANAC AI — SIMPLE ENROLLMENT TEST")
    print("=" * 55)

    # Initialize database
    init_local_db()
    db = LocalSession()

    # Find test student
    student = db.query(Student).filter(
        Student.first_name == 'Chidi',
        Student.last_name == 'Okafor'
    ).first()

    if not student:
        print("❌ Test student not found.")
        print("Run the setup command first.")
        db.close()
        return

    print(f"\n📋 Enrolling: {student.first_name} {student.last_name}")
    print(f"📋 ID: {student.id}")
    print("\n📸 Instructions:")
    print("  1. Press SPACE to capture (5 times)")
    print("  2. Move slightly between captures")
    print("  3. Press ESC to cancel\n")

    # Initialize components
    detector = FaceDetector()
    embedder = FaceEmbedder()
    
    # Open camera with better settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Cannot access camera")
        db.close()
        return

    embeddings = []
    capture_count = 0
    required = 5
    
    print(f"\n📸 Starting capture ({required} required)")
    print("-" * 40)

    while capture_count < required:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame read error, retrying...")
            continue
        
        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Detect face
        face = detector.detect_largest_face(frame)
        
        status_text = "No face detected"
        color = (0, 0, 255)  # Red
        
        if face is not None:
            is_ok, reason = detector.is_face_quality_acceptable(face)
            if is_ok:
                # Draw green box
                x1, y1, x2, y2 = [int(c) for c in face.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                status_text = f"✅ Ready — Press SPACE ({capture_count}/{required})"
                color = (0, 255, 0)
            else:
                status_text = f"⚠️  {reason}"
                color = (0, 165, 255)
                # Draw yellow box
                x1, y1, x2, y2 = [int(c) for c in face.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        # Display info
        cv2.putText(display, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"Captures: {capture_count}/{required}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "ESC to cancel", (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Enrollment - Chidi Okafor", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("\n❌ Enrollment cancelled.")
            break
        
        if key == 32 and face is not None:  # SPACE key
            is_ok, reason = detector.is_face_quality_acceptable(face)
            if is_ok:
                embedding = embedder.extract(face)
                embeddings.append(embedding)
                capture_count += 1
                print(f"  ✅ Capture {capture_count}/{required} saved")
                
                # Flash confirmation
                flash = display.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), 
                              (0, 255, 0), 20)
                cv2.imshow("Enrollment - Chidi Okafor", flash)
                cv2.waitKey(200)
            else:
                print(f"  ❌ Capture rejected: {reason}")

    cap.release()
    cv2.destroyAllWindows()

    # Save to database
    if capture_count >= required:
        print("\n💾 Saving embeddings to database...")
        try:
            # Remove old embeddings
            from database.models import FaceEmbedding
            db.query(FaceEmbedding).filter(
                FaceEmbedding.student_id == student.id
            ).delete()
            
            for embedding in embeddings:
                face_emb = FaceEmbedding(
                    student_id=student.id,
                    embedding=embedder.to_json(embedding),
                    model_name='buffalo_l'
                )
                db.add(face_emb)
            
            # Update student
            student.is_enrolled = True
            db.commit()
            
            print("✅ Enrollment complete!")
            print(f"   Student: {student.first_name} {student.last_name}")
            print(f"   Captures: {capture_count}")
            print(f"   Embeddings stored: {len(embeddings)}")
            
        except Exception as e:
            print(f"❌ Error saving: {e}")
            db.rollback()
    else:
        print(f"\n⚠️  Enrollment incomplete: {capture_count}/{required} captures")

    db.close()
    print("=" * 55)


if __name__ == '__main__':
    test_enrollment_simple()