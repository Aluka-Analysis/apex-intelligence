"""
Almanac AI — FastAPI Backend
Sprint 3: Complete API serving recognition,
enrollment, attendance, and dashboard.
"""

import cv2
import base64
import numpy as np
from datetime import datetime, date
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.connection import init_local_db, get_local_db, LocalSession
from database.models import School, Student, AttendanceRecord, AuditLog
from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder
from ai.recognition.matcher import FaceMatcher
from ai.recognition.liveness import LivenessDetector
from ai.enrollment.enroller import StudentEnroller
from backend.schemas import (
    StudentCreate, StudentResponse,
    RecognizeRequest, RecognizeResponse,
    AttendanceResponse, DailyReportResponse,
    HealthResponse, ModelInfoResponse,
    EnrollmentStatusResponse
)


# ── Application lifespan ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all AI models on startup.
    Models load once and stay in memory.
    """
    print("Almanac AI starting up...")
    init_local_db()

    # Load AI models into app state
    app.state.detector  = FaceDetector()
    app.state.embedder  = FaceEmbedder()
    app.state.matcher   = FaceMatcher(threshold=0.45)
    app.state.liveness  = LivenessDetector(
        ear_threshold   = 0.25,
        blinks_required = 1,
        session_seconds = 10.0
    )
    app.state.enroller  = StudentEnroller()

    # Load enrolled embeddings into memory
    db = LocalSession()
    app.state.enrolled = app.state.enroller.load_all_embeddings(db)
    db.close()

    print(f"Almanac AI ready.")
    print(f"Enrolled students: {len(app.state.enrolled)}")

    yield

    print("Almanac AI shutting down.")


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title       = "Almanac AI",
    description = "Intelligent Attendance Management System by Apex Intelligence",
    version     = "1.0.0",
    lifespan    = lifespan
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Mount frontend ────────────────────────────────────────────
import os
if os.path.exists("frontend"):
    app.mount(
        "/static",
        StaticFiles(directory="frontend"),
        name="static"
    )


# ── Helper functions ──────────────────────────────────────────
def get_or_create_school(db: Session) -> School:
    """Get the first school or create default."""
    school = db.query(School).first()
    if not school:
        school = School(
            name    = "Greenfield Academy",
            address = "Port Harcourt, Rivers State",
            phone   = "08012345678"
        )
        db.add(school)
        db.commit()
    return school


def reload_embeddings():
    """Reload enrolled embeddings after new enrollment."""
    db = LocalSession()
    app.state.enrolled = app.state.enroller.load_all_embeddings(db)
    db.close()


def decode_image(image_base64: str) -> np.ndarray:
    """
    Decode base64 image string to OpenCV frame.
    Frontend sends camera frames as base64 JPEG.
    """
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame       = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode image")

        return frame

    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = f"Invalid image data: {str(e)}"
        )


# ── ROUTES ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the admin dashboard."""
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        with open(frontend_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Almanac AI API Running</h1><p>Visit /docs for API documentation.</p>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    enrolled_count = len(app.state.enrolled)
    return HealthResponse(
        status          = "healthy",
        version         = "1.0.0",
        enrolled_count  = enrolled_count,
        models_loaded   = True,
        timestamp       = datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model and system information."""
    return ModelInfoResponse(
        recognition_model   = "InsightFace buffalo_l",
        embedding_dimension = 512,
        liveness_method     = "Eye Aspect Ratio — Single Blink",
        threshold           = 0.45,
        enrolled_count      = len(app.state.enrolled),
        framework           = "ONNX Runtime CPU"
    )


# ── STUDENT ENDPOINTS ─────────────────────────────────────────

@app.post("/students", response_model=StudentResponse, status_code=201)
async def create_student(
    payload: StudentCreate,
    db:      Session = Depends(get_local_db)
):
    """
    Create a new student record.
    Student is not enrolled until face capture is completed.
    """
    school = get_or_create_school(db)

    # Check for duplicate
    existing = db.query(Student).filter(
        Student.first_name == payload.first_name,
        Student.last_name  == payload.last_name,
        Student.school_id  == school.id
    ).first()

    if existing:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = f"Student {payload.first_name} {payload.last_name} already exists."
        )

    student = Student(
        school_id     = school.id,
        student_id_no = payload.student_id_no,
        first_name    = payload.first_name,
        last_name     = payload.last_name,
        class_name    = payload.class_name,
        parent_phone  = payload.parent_phone  or "",
        parent_email  = payload.parent_email  or "",
        is_enrolled   = False
    )
    db.add(student)
    db.commit()

    # Audit log
    audit = AuditLog(
        action       = "STUDENT_CREATED",
        target_id    = student.id,
        details      = f"{student.first_name} {student.last_name} — {student.class_name}",
        performed_by = "admin"
    )
    db.add(audit)
    db.commit()

    return StudentResponse(
        id            = student.id,
        student_id_no = student.student_id_no,
        first_name    = student.first_name,
        last_name     = student.last_name,
        class_name    = student.class_name,
        parent_phone  = student.parent_phone,
        is_enrolled   = student.is_enrolled,
        created_at    = student.created_at.isoformat() if student.created_at else None
    )


@app.get("/students", response_model=list[StudentResponse])
async def list_students(
    class_name: Optional[str] = None,
    enrolled:   Optional[bool] = None,
    db:         Session = Depends(get_local_db)
):
    """List all students with optional filters."""
    query = db.query(Student).filter(Student.is_active == True)

    if class_name:
        query = query.filter(Student.class_name == class_name)

    if enrolled is not None:
        query = query.filter(Student.is_enrolled == enrolled)

    students = query.order_by(Student.last_name).all()

    return [
        StudentResponse(
            id            = s.id,
            student_id_no = s.student_id_no,
            first_name    = s.first_name,
            last_name     = s.last_name,
            class_name    = s.class_name,
            parent_phone  = s.parent_phone,
            is_enrolled   = s.is_enrolled,
            created_at    = s.created_at.isoformat() if s.created_at else None
        )
        for s in students
    ]


@app.get("/students/{student_id}", response_model=StudentResponse)
async def get_student(
    student_id: str,
    db:         Session = Depends(get_local_db)
):
    """Get a single student by ID."""
    student = db.query(Student).filter(
        Student.id        == student_id,
        Student.is_active == True
    ).first()

    if not student:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "Student not found"
        )

    return StudentResponse(
        id            = student.id,
        student_id_no = student.student_id_no,
        first_name    = student.first_name,
        last_name     = student.last_name,
        class_name    = student.class_name,
        parent_phone  = student.parent_phone,
        is_enrolled   = student.is_enrolled,
        created_at    = student.created_at.isoformat() if student.created_at else None
    )


# ── ENROLLMENT ENDPOINT ───────────────────────────────────────

@app.post("/students/{student_id}/enroll")
async def enroll_face(
    student_id:    str,
    payload:       dict,
    db:            Session = Depends(get_local_db)
):
    """
    Enroll a single face image for a student.
    Frontend sends multiple captures one by one.
    When 5 captures are received enrollment is complete.

    Payload:
        image_base64: base64 encoded face image
    """
    student = db.query(Student).filter(
        Student.id == student_id
    ).first()

    if not student:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "Student not found"
        )

    image_b64 = payload.get("image_base64")
    if not image_b64:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "image_base64 is required"
        )

    frame = decode_image(image_b64)
    face  = app.state.detector.detect_largest_face(frame)

    if face is None:
        return {
            "success":  False,
            "message":  "No face detected in image",
            "captures": 0
        }

    is_ok, reason = app.state.detector.is_face_quality_acceptable(face)
    if not is_ok:
        return {
            "success":  False,
            "message":  f"Face quality insufficient: {reason}",
            "captures": 0
        }

    # Save embedding
    from database.models import FaceEmbedding
    embedding = app.state.embedder.extract(face)

    face_emb = FaceEmbedding(
        student_id = student_id,
        embedding  = app.state.embedder.to_json(embedding),
        model_name = "buffalo_l"
    )
    db.add(face_emb)
    db.commit()

    # Count total captures
    total_captures = db.execute(
        text("SELECT COUNT(*) FROM face_embeddings WHERE student_id = :sid"),
        {"sid": student_id}
    ).scalar()

    # Mark enrolled when 5 captures reached
    if total_captures >= 5 and not student.is_enrolled:
        student.is_enrolled = True
        db.commit()

        audit = AuditLog(
            action       = "STUDENT_ENROLLED",
            target_id    = student_id,
            details      = f"{student.first_name} {student.last_name} — {total_captures} captures",
            performed_by = "admin"
        )
        db.add(audit)
        db.commit()

        reload_embeddings()

        return {
            "success":         True,
            "message":         f"{student.first_name} {student.last_name} enrolled successfully",
            "captures":        total_captures,
            "enrollment_done": True
        }

    return {
        "success":         True,
        "message":         f"Capture {total_captures}/5 saved",
        "captures":        total_captures,
        "enrollment_done": False
    }


# ── RECOGNITION ENDPOINT ──────────────────────────────────────

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_face(
    payload: RecognizeRequest,
    db:      Session = Depends(get_local_db)
):
    """
    Identify a face in a camera frame.
    Called repeatedly from the frontend camera feed.
    Returns identity match and liveness status.
    """
    if not app.state.enrolled:
        return RecognizeResponse(
            matched        = False,
            student_id     = None,
            student_name   = None,
            class_name     = None,
            confidence     = 0.0,
            liveness_passed = False,
            liveness_status = "No enrolled students",
            attendance_logged = False,
            message        = "No enrolled students found"
        )

    frame = decode_image(payload.image_base64)
    face  = app.state.detector.detect_largest_face(frame)

    if face is None:
        return RecognizeResponse(
            matched           = False,
            student_id        = None,
            student_name      = None,
            class_name        = None,
            confidence        = 0.0,
            liveness_passed   = False,
            liveness_status   = "No face detected",
            attendance_logged = False,
            message           = "No face detected in frame"
        )

    # Identity matching
    embedding    = app.state.embedder.extract(face)
    match_result = app.state.matcher.match(embedding, app.state.enrolled)

    if not match_result["matched"]:
        return RecognizeResponse(
            matched           = False,
            student_id        = None,
            student_name      = None,
            class_name        = None,
            confidence        = match_result["confidence"],
            liveness_passed   = False,
            liveness_status   = "Unknown face",
            attendance_logged = False,
            message           = f"Unknown face — best score {match_result['confidence']:.1%}"
        )

    student_id   = match_result["student_id"]
    student_name = match_result["student_name"]
    class_name   = match_result["class_name"]
    confidence   = match_result["confidence"]

    # Liveness check
    live_result = app.state.liveness.update(face, student_id)

    if not live_result["is_live"]:
        return RecognizeResponse(
            matched           = True,
            student_id        = student_id,
            student_name      = student_name,
            class_name        = class_name,
            confidence        = confidence,
            liveness_passed   = False,
            liveness_status   = live_result["status"],
            attendance_logged = False,
            message           = live_result["status"]
        )

    # Check if already logged today
    today = datetime.now().strftime("%Y-%m-%d")

    already_logged = db.query(AttendanceRecord).filter(
        AttendanceRecord.student_id == student_id,
        AttendanceRecord.date       == today
    ).first()

    if already_logged:
        return RecognizeResponse(
            matched           = True,
            student_id        = student_id,
            student_name      = student_name,
            class_name        = class_name,
            confidence        = confidence,
            liveness_passed   = True,
            liveness_status   = "Liveness confirmed",
            attendance_logged = True,
            message           = f"{student_name} already logged today at {already_logged.time_recorded.strftime('%H:%M:%S')}"
        )

    # Log attendance
    school_id = db.execute(
        text("SELECT school_id FROM students WHERE id = :sid"),
        {"sid": student_id}
    ).scalar()

    record = AttendanceRecord(
        student_id  = student_id,
        school_id   = school_id,
        date        = today,
        status      = "present",
        confidence  = confidence,
        method      = "face_recognition"
    )
    db.add(record)
    db.commit()

    return RecognizeResponse(
        matched           = True,
        student_id        = student_id,
        student_name      = student_name,
        class_name        = class_name,
        confidence        = confidence,
        liveness_passed   = True,
        liveness_status   = "Liveness confirmed",
        attendance_logged = True,
        message           = f"Attendance logged for {student_name} at {datetime.now().strftime('%H:%M:%S')}"
    )


# ── ATTENDANCE ENDPOINTS ──────────────────────────────────────

@app.get("/attendance/{date_str}", response_model=DailyReportResponse)
async def get_daily_attendance(
    date_str: str,
    db:       Session = Depends(get_local_db)
):
    """
    Get attendance report for a specific date.
    date_str format: YYYY-MM-DD
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Invalid date format. Use YYYY-MM-DD"
        )

    records = db.execute(text("""
        SELECT
            s.id,
            s.first_name,
            s.last_name,
            s.class_name,
            a.status,
            a.time_recorded,
            a.confidence
        FROM attendance_records a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = :date
        ORDER BY a.time_recorded ASC
    """), {"date": date_str}).fetchall()

    total_students = db.query(Student).filter(
        Student.is_enrolled == True,
        Student.is_active   == True
    ).count()

    present_count = len(records)
    absent_count  = total_students - present_count

    attendance_list = [
        AttendanceResponse(
            student_id   = str(r[0]),
            first_name   = r[1],
            last_name    = r[2],
            class_name   = r[3],
            status       = r[4],
            time_recorded = str(r[5])[11:19] if r[5] and len(str(r[5])) > 10 else str(r[5])[:8] if r[5] else None,
            confidence   = round(r[6], 4) if r[6] else None
        )
        for r in records
    ]

    return DailyReportResponse(
        date           = date_str,
        total_students = total_students,
        present        = present_count,
        absent         = absent_count,
        attendance_rate = round(present_count / total_students * 100, 1) if total_students > 0 else 0,
        records        = attendance_list
    )


@app.get("/attendance/student/{student_id}")
async def get_student_attendance_history(
    student_id: str,
    limit:      int     = 30,
    db:         Session = Depends(get_local_db)
):
    """Get attendance history for a specific student."""
    student = db.query(Student).filter(
        Student.id == student_id
    ).first()

    if not student:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "Student not found"
        )

    records = db.query(AttendanceRecord).filter(
        AttendanceRecord.student_id == student_id
    ).order_by(
        AttendanceRecord.date.desc()
    ).limit(limit).all()

    return {
        "student_id":   student_id,
        "student_name": f"{student.first_name} {student.last_name}",
        "class_name":   student.class_name,
        "total_days":   len(records),
        "records": [
            {
                "date":          r.date,
                "status":        r.status,
                "time_recorded": str(r.time_recorded)[11:19] if r.time_recorded else None,
                "confidence":    round(r.confidence, 4) if r.confidence else None
            }
            for r in records
        ]
    }


@app.get("/dashboard/summary")
async def dashboard_summary(db: Session = Depends(get_local_db)):
    """
    Summary statistics for the admin dashboard.
    Returns today's attendance overview.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    total_enrolled = db.query(Student).filter(
        Student.is_enrolled == True,
        Student.is_active   == True
    ).count()

    total_present = db.query(AttendanceRecord).filter(
        AttendanceRecord.date == today
    ).count()

    total_students = db.query(Student).filter(
        Student.is_active == True
    ).count()

    # Class breakdown
    class_data = db.execute(text("""
        SELECT
            s.class_name,
            COUNT(s.id) as total,
            SUM(CASE WHEN s.is_enrolled THEN 1 ELSE 0 END) as enrolled,
            COUNT(a.id) as present_today
        FROM students s
        LEFT JOIN attendance_records a
            ON s.id = a.student_id AND a.date = :today
        WHERE s.is_active = 1
        GROUP BY s.class_name
        ORDER BY s.class_name
    """), {"today": today}).fetchall()

    return {
        "date":             today,
        "day":              datetime.now().strftime("%A"),
        "total_students":   total_students,
        "total_enrolled":   total_enrolled,
        "present_today":    total_present,
        "absent_today":     total_enrolled - total_present,
        "attendance_rate":  round(total_present / total_enrolled * 100, 1) if total_enrolled > 0 else 0,
        "enrolled_count":   len(app.state.enrolled),
        "class_breakdown": [
            {
                "class_name":    r[0],
                "total":         r[1],
                "enrolled":      r[2],
                "present_today": r[3]
            }
            for r in class_data
        ]
    }

# ── SESSION MANAGEMENT ────────────────────────────────────────

@app.post("/session/start")
async def start_school_session(
    db: Session = Depends(get_local_db)
):
    """
    Mark the start of the school day.
    Opens attendance window for today.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")

    audit = AuditLog(
        action       = "SESSION_STARTED",
        target_id    = today,
        details      = f"School day started at {now}",
        performed_by = "admin"
    )
    db.add(audit)
    db.commit()

    return {
        "status":  "session_started",
        "date":    today,
        "time":    now,
        "message": f"School day started at {now}. Attendance window is open."
    }


@app.post("/session/end")
async def end_school_session(
    db: Session = Depends(get_local_db)
):
    """
    Close the school day.
    Marks all enrolled students not yet seen as absent.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")

    # Find enrolled students not yet logged today
    school = get_or_create_school(db)

    enrolled_students = db.query(Student).filter(
        Student.is_enrolled == True,
        Student.is_active   == True,
        Student.school_id   == school.id
    ).all()

    present_today = db.execute(
        text("SELECT student_id FROM attendance_records WHERE date = :date"),
        {"date": today}
    ).fetchall()

    present_ids = {r[0] for r in present_today}
    absent_count = 0

    for student in enrolled_students:
        if student.id not in present_ids:
            record = AttendanceRecord(
                student_id  = student.id,
                school_id   = school.id,
                date        = today,
                status      = "absent",
                confidence  = None,
                method      = "system_end_of_day"
            )
            db.add(record)
            absent_count += 1

    db.commit()

    audit = AuditLog(
        action       = "SESSION_ENDED",
        target_id    = today,
        details      = f"School day ended at {now}. {absent_count} students marked absent.",
        performed_by = "admin"
    )
    db.add(audit)
    db.commit()

    return {
        "status":        "session_ended",
        "date":          today,
        "time":          now,
        "absent_marked": absent_count,
        "message":       f"School day ended. {absent_count} students marked absent."
    }


@app.get("/session/status")
async def session_status(db: Session = Depends(get_local_db)):
    """
    Get current session status for today.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    start_log = db.query(AuditLog).filter(
        AuditLog.action    == "SESSION_STARTED",
        AuditLog.target_id == today
    ).first()

    end_log = db.query(AuditLog).filter(
        AuditLog.action    == "SESSION_ENDED",
        AuditLog.target_id == today
    ).first()

    present_count = db.query(AttendanceRecord).filter(
        AttendanceRecord.date   == today,
        AttendanceRecord.status == "present"
    ).count()

    return {
        "date":          today,
        "session_started": start_log is not None,
        "session_ended":   end_log is not None,
        "start_time":    start_log.timestamp.strftime("%H:%M:%S") if start_log and start_log.timestamp else None,
        "end_time":      end_log.timestamp.strftime("%H:%M:%S")   if end_log   and end_log.timestamp   else None,
        "present_count": present_count,
        "is_active":     start_log is not None and end_log is None
    }