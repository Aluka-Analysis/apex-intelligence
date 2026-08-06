cat > main.py << 'EOF'
import os
import cv2
import numpy as np
import json
import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from insightface.app import FaceAnalysis
from config import Config
import sqlite3

# Initialize FastAPI
app = FastAPI(
    title="Apex Intelligence - Face Attendance System",
    description="AI-powered face recognition attendance system",
    version="1.0.0"
)

# Initialize InsightFace
face_app = FaceAnalysis(name=Config.INSIGHTFACE_MODEL)
face_app.prepare(ctx_id=0, det_size=Config.FACE_DETECTION_SIZE)

# Database setup
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            face_embedding TEXT NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'present'
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Models
class StudentRegister(BaseModel):
    name: str
    student_id: str
    image: str  # base64 encoded image

class AttendanceRequest(BaseModel):
    image: str  # base64 encoded image

# Helper functions
def get_face_embedding(image_bytes):
    """Extract face embedding from image"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    
    return faces[0].embedding.tolist()

def find_student(embedding, threshold=Config.RECOGNITION_THRESHOLD):
    """Find student by face embedding"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, student_id, name, face_embedding FROM students')
    students = cursor.fetchall()
    conn.close()
    
    best_match = None
    best_distance = float('inf')
    
    for student in students:
        stored_embedding = json.loads(student[3])
        distance = 1 - np.dot(embedding, stored_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
        )
        
        if distance < best_distance:
            best_distance = distance
            best_match = student
    
    if best_distance < threshold:
        return {
            'id': best_match[0],
            'student_id': best_match[1],
            'name': best_match[2]
        }
    return None

# API Endpoints
@app.get("/")
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Apex Intelligence - Face Attendance</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
            h1 { color: #1a73e8; }
            .card { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            input, button { padding: 10px; margin: 5px; }
            button { background: #1a73e8; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #1557b0; }
            #video { width: 100%; max-width: 640px; border-radius: 10px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #1a73e8; color: white; }
        </style>
    </head>
    <body>
        <h1>📸 Apex Intelligence - Face Attendance</h1>
        
        <div class="card">
            <h2>📝 Register Student</h2>
            <input type="text" id="name" placeholder="Student Name">
            <input type="text" id="student_id" placeholder="Student ID">
            <button onclick="registerStudent()">Register</button>
            <div id="register_result"></div>
        </div>
        
        <div class="card">
            <h2>✅ Mark Attendance</h2>
            <video id="video" autoplay></video>
            <button onclick="captureAttendance()">📸 Capture & Mark Attendance</button>
            <div id="attendance_result"></div>
        </div>
        
        <div class="card">
            <h2>📋 Recent Attendance</h2>
            <button onclick="getAttendance()">Refresh</button>
            <div id="attendance_list"></div>
        </div>
        
        <script>
            const video = document.getElementById('video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => alert('Camera access denied!'));
            
            async function registerStudent() {
                const name = document.getElementById('name').value;
                const student_id = document.getElementById('student_id').value;
                
                if (!name || !student_id) {
                    alert('Please enter name and student ID');
                    return;
                }
                
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                const image_data = canvas.toDataURL('image/jpeg');
                
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        name: name, 
                        student_id: student_id,
                        image: image_data
                    })
                });
                
                const result = await response.json();
                document.getElementById('register_result').innerHTML = 
                    `<p>${result.message || result.detail}</p>`;
            }
            
            async function captureAttendance() {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                const image_data = canvas.toDataURL('image/jpeg');
                
                const response = await fetch('/attendance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: image_data })
                });
                
                const result = await response.json();
                document.getElementById('attendance_result').innerHTML = 
                    `<p>${result.message || result.detail}</p>`;
                getAttendance();
            }
            
            async function getAttendance() {
                const response = await fetch('/attendance');
                const data = await response.json();
                let html = '<table><tr><th>Name</th><th>Student ID</th><th>Time</th></tr>';
                data.forEach(record => {
                    html += `<tr><td>${record.name}</td><td>${record.student_id}</td><td>${record.timestamp}</td></tr>`;
                });
                html += '</table>';
                document.getElementById('attendance_list').innerHTML = html;
            }
            
            getAttendance();
        </script>
    </body>
    </html>
    """

@app.post("/register")
async def register_student(data: StudentRegister):
    try:
        # Decode image
        image_data = data.image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        embedding = get_face_embedding(image_bytes)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO students (name, student_id, face_embedding) VALUES (?, ?, ?)',
            (data.name, data.student_id, json.dumps(embedding))
        )
        conn.commit()
        conn.close()
        
        return {"message": f"✅ Student {data.name} registered successfully!"}
    
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Student ID already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attendance")
async def mark_attendance(data: AttendanceRequest):
    try:
        image_data = data.image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        embedding = get_face_embedding(image_bytes)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")
        
        student = find_student(embedding)
        if student is None:
            return {"message": "❌ Face not recognized. Please register first."}
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO attendance (student_id, name) VALUES (?, ?)',
            (student['student_id'], student['name'])
        )
        conn.commit()
        conn.close()
        
        return {
            "message": f"✅ Attendance marked for {student['name']} (ID: {student['student_id']})"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attendance")
async def get_attendance():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT student_id, name, timestamp FROM attendance ORDER BY timestamp DESC LIMIT 50'
    )
    records = cursor.fetchall()
    conn.close()
    
    return [
        {
            'student_id': r[0],
            'name': r[1],
            'timestamp': r[2]
        }
        for r in records
    ]

@app.get("/students")
async def get_students():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, student_id, registered_at FROM students')
    students = cursor.fetchall()
    conn.close()
    
    return [
        {
            'id': s[0],
            'name': s[1],
            'student_id': s[2],
            'registered_at': s[3]
        }
        for s in students
    ]

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
EOF