"""
Almanac AI — Complete Web Dashboard
FastAPI application with face enrollment and attendance tracking.
"""

import os
import cv2
import numpy as np
import base64
import json
import uuid
from datetime import datetime, date
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from sqlalchemy.orm import Session
from typing import Optional

from database.connection import init_local_db, get_local_db
from database.models import Student, School, AttendanceRecord, FaceEmbedding, AuditLog
from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder
from ai.recognition.matcher import FaceMatcher
from ai.enrollment.enroller import StudentEnroller

# Initialize FastAPI
app = FastAPI(
    title="Almanac AI — School Attendance System",
    description="Face recognition attendance system with web dashboard",
    version="1.0.0"
)

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize database
init_local_db()

# Initialize face recognition components
detector = FaceDetector()
embedder = FaceEmbedder()
matcher = FaceMatcher(threshold=0.45)
enroller = StudentEnroller(required_captures=5)


def get_student_name(db: Session, student_id: str) -> str:
    """Helper function to get student name from ID"""
    student = db.query(Student).filter(Student.id == student_id).first()
    if student:
        return f"{student.first_name} {student.last_name}"
    return "Unknown Student"


# ==========================================
# WEB PAGES
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_local_db)):
    """Main dashboard page"""
    
    # Get statistics
    student_count = db.query(Student).filter(Student.is_active == True).count()
    enrolled_count = db.query(Student).filter(Student.is_enrolled == True).count()
    attendance_today = db.query(AttendanceRecord).filter(
        AttendanceRecord.date == date.today().isoformat()
    ).count()
    
    # Get recent attendance
    recent = db.query(AttendanceRecord).order_by(
        AttendanceRecord.time_recorded.desc()
    ).limit(10).all()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Almanac AI — Dashboard</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; }}
            .header {{ background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 20px; }}
            .header h1 {{ font-size: 24px; }}
            .header small {{ opacity: 0.8; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
            .stat-number {{ font-size: 32px; font-weight: bold; color: #1a73e8; }}
            .stat-label {{ color: #666; margin-top: 5px; }}
            .nav {{ display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }}
            .nav a {{ background: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; color: #1a73e8; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .nav a:hover {{ background: #1a73e8; color: white; }}
            .table-container {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background: #f8f9fa; color: #333; }}
            .status-present {{ color: #4caf50; font-weight: bold; }}
            .status-absent {{ color: #f44336; font-weight: bold; }}
            .btn {{ background: #1a73e8; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
            .btn:hover {{ background: #1557b0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>🏫 Almanac AI — Attendance System</h1>
                <small>Face recognition powered school attendance</small>
            </div>
        </div>
        
        <div class="container">
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{student_count}</div>
                    <div class="stat-label">Total Students</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{enrolled_count}</div>
                    <div class="stat-label">Face Enrolled</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{attendance_today}</div>
                    <div class="stat-label">Today's Attendance</div>
                </div>
            </div>
            
            <div class="nav">
                <a href="/">📊 Dashboard</a>
                <a href="/enroll">📸 Enroll Student</a>
                <a href="/attendance">📋 Attendance</a>
                <a href="/students">👨‍🎓 Students</a>
                <a href="/mark">✅ Mark Attendance</a>
            </div>
            
            <div class="table-container">
                <h3>Recent Attendance</h3>
                <table>
                    <tr>
                        <th>Student</th>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Confidence</th>
                        <th>Status</th>
                    </tr>
    """
    
    for record in recent:
        status_class = "status-present" if record.status == "present" else "status-absent"
        confidence = f"{record.confidence*100:.1f}%" if record.confidence else "N/A"
        student_name = get_student_name(db, record.student_id)
        html += f"""
                    <tr>
                        <td>{student_name}</td>
                        <td>{record.date}</td>
                        <td>{record.time_recorded.strftime('%H:%M:%S')}</td>
                        <td>{confidence}</td>
                        <td class="{status_class}">{record.status.upper()}</td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


@app.get("/enroll", response_class=HTMLResponse)
async def enroll_page(request: Request, db: Session = Depends(get_local_db)):
    """Student enrollment page"""
    
    students = db.query(Student).filter(
        Student.is_active == True,
        Student.is_enrolled == False
    ).all()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enroll Student — Almanac AI</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; }}
            .header {{ background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 20px; }}
            .header h1 {{ font-size: 24px; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            .card {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
            .form-group {{ margin: 15px 0; }}
            label {{ display: block; margin-bottom: 5px; font-weight: 500; color: #333; }}
            input, select {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }}
            .btn {{ background: #1a73e8; color: white; border: none; padding: 12px 30px; border-radius: 5px; cursor: pointer; font-size: 16px; }}
            .btn:hover {{ background: #1557b0; }}
            .btn-success {{ background: #4caf50; }}
            .btn-success:hover {{ background: #388e3c; }}
            .nav {{ display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }}
            .nav a {{ background: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; color: #1a73e8; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .nav a:hover {{ background: #1a73e8; color: white; }}
            #video {{ width: 100%; max-width: 640px; border-radius: 10px; background: #333; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .status-success {{ background: #d4edda; color: #155724; }}
            .status-error {{ background: #f8d7da; color: #721c24; }}
            .status-info {{ background: #d1ecf1; color: #0c5460; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>📸 Enroll Student</h1>
            </div>
        </div>
        
        <div class="container">
            <div class="nav">
                <a href="/">📊 Dashboard</a>
                <a href="/enroll">📸 Enroll Student</a>
                <a href="/attendance">📋 Attendance</a>
                <a href="/students">👨‍🎓 Students</a>
                <a href="/mark">✅ Mark Attendance</a>
            </div>
            
            <div class="card">
                <h3>Select Student</h3>
                <div class="form-group">
                    <label>Student</label>
                    <select id="student_id">
    """
    
    for student in students:
        html += f'<option value="{student.id}">{student.first_name} {student.last_name} ({student.student_id_no})</option>'
    
    if not students:
        html += '<option value="">No students available</option>'
    
    html += f"""
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Video Feed</label>
                    <video id="video" autoplay></video>
                    <div id="status" class="status status-info">📸 Position your face in the camera</div>
                </div>
                
                <button class="btn btn-success" onclick="startEnrollment()">🎯 Start Enrollment</button>
                <button class="btn" onclick="cancelEnrollment()">❌ Cancel</button>
                
                <div id="result"></div>
            </div>
        </div>
        
        <script>
            const video = document.getElementById('video');
            let stream = null;
            let isEnrolling = false;
            
            navigator.mediaDevices.getUserMedia({{ video: true }})
                .then(s => {{
                    stream = s;
                    video.srcObject = s;
                }})
                .catch(err => {{
                    document.getElementById('status').innerHTML = '❌ Camera access denied';
                    document.getElementById('status').className = 'status status-error';
                }});
            
            async function startEnrollment() {{
                if (isEnrolling) return;
                isEnrolling = true;
                
                const studentId = document.getElementById('student_id').value;
                if (!studentId) {{
                    alert('Please select a student');
                    isEnrolling = false;
                    return;
                }}
                
                const captures = [];
                const required = 5;
                const status = document.getElementById('status');
                
                for (let i = 0; i < required; i++) {{
                    status.innerHTML = `📸 Capturing ${{i+1}}/${{required}} - Look at the camera`;
                    status.className = 'status status-info';
                    
                    await new Promise(r => setTimeout(r, 800));
                    
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    const imageData = canvas.toDataURL('image/jpeg');
                    captures.push(imageData);
                }}
                
                status.innerHTML = '💾 Saving enrollment...';
                status.className = 'status status-info';
                
                const response = await fetch('/api/enroll', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        student_id: studentId,
                        captures: captures
                    }})
                }});
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                
                if (result.success) {{
                    resultDiv.innerHTML = `<div class="status status-success">✅ ${{result.message}}</div>`;
                    status.innerHTML = '✅ Enrollment complete!';
                    status.className = 'status status-success';
                }} else {{
                    resultDiv.innerHTML = `<div class="status status-error">❌ ${{result.message}}</div>`;
                    status.innerHTML = '❌ Enrollment failed';
                    status.className = 'status status-error';
                }}
                
                isEnrolling = false;
            }}
            
            function cancelEnrollment() {{
                if (stream) {{
                    stream.getTracks().forEach(t => t.stop());
                }}
                document.getElementById('status').innerHTML = '❌ Enrollment cancelled';
                document.getElementById('status').className = 'status status-error';
                isEnrolling = false;
            }}
        </script>
    </body>
    </html>
    """
    
    return html


@app.get("/attendance", response_class=HTMLResponse)
async def attendance_page(request: Request, db: Session = Depends(get_local_db)):
    """Attendance records page"""
    
    records = db.query(AttendanceRecord).order_by(
        AttendanceRecord.time_recorded.desc()
    ).limit(50).all()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attendance — Almanac AI</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; }
            .header { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 20px; }
            .header h1 { font-size: 24px; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .nav { display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }
            .nav a { background: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; color: #1a73e8; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .nav a:hover { background: #1a73e8; color: white; }
            .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; color: #333; }
            .status-present { color: #4caf50; font-weight: bold; }
            .status-absent { color: #f44336; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>📋 Attendance Records</h1>
            </div>
        </div>
        
        <div class="container">
            <div class="nav">
                <a href="/">📊 Dashboard</a>
                <a href="/enroll">📸 Enroll Student</a>
                <a href="/attendance">📋 Attendance</a>
                <a href="/students">👨‍🎓 Students</a>
                <a href="/mark">✅ Mark Attendance</a>
            </div>
            
            <div class="card">
                <h3>Recent Attendance (Last 50)</h3>
                <table>
                    <tr>
                        <th>Student</th>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Confidence</th>
                        <th>Status</th>
                    </tr>
    """
    
    for record in records:
        status_class = "status-present" if record.status == "present" else "status-absent"
        confidence = f"{record.confidence*100:.1f}%" if record.confidence else "N/A"
        student_name = get_student_name(db, record.student_id)
        html += f"""
                    <tr>
                        <td>{student_name}</td>
                        <td>{record.date}</td>
                        <td>{record.time_recorded.strftime('%H:%M:%S')}</td>
                        <td>{confidence}</td>
                        <td class="{status_class}">{record.status.upper()}</td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


@app.get("/students", response_class=HTMLResponse)
async def students_page(request: Request, db: Session = Depends(get_local_db)):
    """Students management page"""
    
    students = db.query(Student).filter(Student.is_active == True).all()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Students — Almanac AI</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; }
            .header { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 20px; }
            .header h1 { font-size: 24px; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .nav { display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }
            .nav a { background: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; color: #1a73e8; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .nav a:hover { background: #1a73e8; color: white; }
            .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; color: #333; }
            .badge-enrolled { background: #4caf50; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; }
            .badge-not-enrolled { background: #ff9800; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; }
            .btn { background: #1a73e8; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }
            .btn:hover { background: #1557b0; }
            .btn-success { background: #4caf50; }
            .btn-success:hover { background: #388e3c; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>👨‍🎓 Students</h1>
            </div>
        </div>
        
        <div class="container">
            <div class="nav">
                <a href="/">📊 Dashboard</a>
                <a href="/enroll">📸 Enroll Student</a>
                <a href="/attendance">📋 Attendance</a>
                <a href="/students">👨‍🎓 Students</a>
                <a href="/mark">✅ Mark Attendance</a>
            </div>
            
            <div class="card">
                <h3>All Students</h3>
                <br>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Class</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
    """
    
    for student in students:
        status_badge = "badge-enrolled" if student.is_enrolled else "badge-not-enrolled"
        status_text = "✅ Enrolled" if student.is_enrolled else "⏳ Pending"
        html += f"""
                    <tr>
                        <td>{student.student_id_no}</td>
                        <td>{student.first_name} {student.last_name}</td>
                        <td>{student.class_name or 'N/A'}</td>
                        <td><span class="{status_badge}">{status_text}</span></td>
                        <td>
                            <a href="/enroll" class="btn" style="padding:5px 15px; font-size:12px;">Enroll</a>
                        </td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


@app.get("/mark", response_class=HTMLResponse)
async def mark_page(request: Request):
    """Mark attendance page"""
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mark Attendance — Almanac AI</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; }
            .header { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 20px; }
            .header h1 { font-size: 24px; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .nav { display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }
            .nav a { background: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; color: #1a73e8; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .nav a:hover { background: #1a73e8; color: white; }
            .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
            #video { width: 100%; max-width: 640px; border-radius: 10px; background: #333; margin: 20px 0; }
            .btn { background: #1a73e8; color: white; border: none; padding: 15px 30px; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .btn:hover { background: #1557b0; }
            .btn-success { background: #4caf50; }
            .btn-success:hover { background: #388e3c; }
            .status { padding: 15px; border-radius: 5px; margin: 15px 0; font-size: 18px; }
            .status-success { background: #d4edda; color: #155724; }
            .status-error { background: #f8d7da; color: #721c24; }
            .status-info { background: #d1ecf1; color: #0c5460; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>✅ Mark Attendance</h1>
            </div>
        </div>
        
        <div class="container">
            <div class="nav">
                <a href="/">📊 Dashboard</a>
                <a href="/enroll">📸 Enroll Student</a>
                <a href="/attendance">📋 Attendance</a>
                <a href="/students">👨‍🎓 Students</a>
                <a href="/mark">✅ Mark Attendance</a>
            </div>
            
            <div class="card">
                <h3>Face Recognition Attendance</h3>
                <video id="video" autoplay></video>
                <div id="status" class="status status-info">📸 Looking for faces...</div>
                <button class="btn btn-success" onclick="markAttendance()">📸 Mark Attendance</button>
                <div id="result"></div>
            </div>
        </div>
        
        <script>
            const video = document.getElementById('video');
            let stream = null;
            
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(s => { stream = s; video.srcObject = s; })
                .catch(err => { document.getElementById('status').innerHTML = '❌ Camera access denied'; });
            
            async function markAttendance() {
                const status = document.getElementById('status');
                const result = document.getElementById('result');
                
                status.innerHTML = '📸 Capturing...';
                status.className = 'status status-info';
                
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                const imageData = canvas.toDataURL('image/jpeg');
                
                const response = await fetch('/api/mark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    status.innerHTML = '✅ ' + data.message;
                    status.className = 'status status-success';
                    result.innerHTML = `
                        <div class="status status-success">
                            ✅ ${data.student_name} — ${data.confidence}% confidence<br>
                            Time: ${data.time}
                        </div>
                    `;
                } else {
                    status.innerHTML = '❌ ' + data.message;
                    status.className = 'status status-error';
                    result.innerHTML = `<div class="status status-error">❌ ${data.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """


# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/api/enroll")
async def api_enroll(data: dict, db: Session = Depends(get_local_db)):
    """API endpoint for student enrollment"""
    
    try:
        student_id = data.get('student_id')
        captures = data.get('captures', [])
        
        if not student_id or len(captures) < 5:
            return {'success': False, 'message': 'Invalid enrollment data'}
        
        student = db.query(Student).filter(Student.id == student_id).first()
        if not student:
            return {'success': False, 'message': 'Student not found'}
        
        embeddings = []
        detector = FaceDetector()
        embedder = FaceEmbedder()
        
        for cap in captures:
            if ',' in cap:
                cap = cap.split(',')[1]
            image_bytes = base64.b64decode(cap)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            face = detector.detect_largest_face(img)
            if face is None:
                continue
            
            is_ok, _ = detector.is_face_quality_acceptable(face)
            if not is_ok:
                continue
            
            embedding = embedder.extract(face)
            embeddings.append(embedding)
        
        if len(embeddings) < 3:
            return {'success': False, 'message': f'Only {len(embeddings)} valid captures. Need at least 3.'}
        
        db.query(FaceEmbedding).filter(FaceEmbedding.student_id == student_id).delete()
        
        for embedding in embeddings:
            face_emb = FaceEmbedding(
                student_id=student_id,
                embedding=embedder.to_json(embedding),
                model_name='buffalo_l'
            )
            db.add(face_emb)
        
        student.is_enrolled = True
        db.commit()
        
        audit = AuditLog(
            action='STUDENT_ENROLLED',
            target_id=student_id,
            details=json.dumps({
                'student_name': f"{student.first_name} {student.last_name}",
                'captures': len(embeddings),
                'model': 'buffalo_l'
            }),
            performed_by='web_dashboard'
        )
        db.add(audit)
        db.commit()
        
        return {
            'success': True,
            'message': f'{student.first_name} {student.last_name} enrolled with {len(embeddings)} captures'
        }
        
    except Exception as e:
        db.rollback()
        return {'success': False, 'message': str(e)}


@app.post("/api/mark")
async def api_mark(data: dict, db: Session = Depends(get_local_db)):
    """API endpoint for marking attendance"""
    
    try:
        image_data = data.get('image', '')
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        face = detector.detect_largest_face(img)
        if face is None:
            return {'success': False, 'message': 'No face detected'}
        
        is_ok, _ = detector.is_face_quality_acceptable(face)
        if not is_ok:
            return {'success': False, 'message': 'Face quality too low'}
        
        embedding = embedder.extract(face)
        enrolled = enroller.load_all_embeddings(db)
        result = matcher.match(embedding, enrolled)
        
        if not result['matched']:
            return {'success': False, 'message': f'Unknown face (confidence: {result["confidence"]:.1%})'}
        
        now = datetime.now()
        record = AttendanceRecord(
            student_id=result['student_id'],
            school_id=db.query(Student).filter(Student.id == result['student_id']).first().school_id,
            date=now.strftime('%Y-%m-%d'),
            time_recorded=now,
            status='present',
            confidence=result['confidence'],
            method='face_recognition'
        )
        db.add(record)
        db.commit()
        
        return {
            'success': True,
            'student_name': result['student_name'],
            'confidence': f"{result['confidence']*100:.1f}",
            'time': now.strftime('%H:%M:%S'),
            'message': f"Attendance marked for {result['student_name']}"
        }
        
    except Exception as e:
        return {'success': False, 'message': str(e)}


# ==========================================
# RUN THE APPLICATION
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)