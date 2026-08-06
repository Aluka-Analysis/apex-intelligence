"""
Database models for Almanac AI.
Defines all tables using SQLAlchemy ORM.
"""

from sqlalchemy import (
    Column, String, Integer, Float,
    DateTime, Boolean, Text, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class School(Base):
    """
    Represents a school using Almanac AI.
    """
    __tablename__ = 'schools'

    id          = Column(String, primary_key=True, default=generate_uuid)
    name        = Column(String(255), nullable=False)
    address     = Column(Text, nullable=True)
    phone       = Column(String(20), nullable=True)
    email       = Column(String(255), nullable=True)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, server_default=func.now())
    updated_at  = Column(DateTime, onupdate=func.now())

    def __repr__(self):
        return f"<School {self.name}>"


class Student(Base):
    """
    Represents a student enrolled in the system.
    """
    __tablename__ = 'students'

    id              = Column(String, primary_key=True, default=generate_uuid)
    school_id       = Column(String, ForeignKey('schools.id'), nullable=False)
    student_id_no   = Column(String(50), nullable=False)
    first_name      = Column(String(100), nullable=False)
    last_name       = Column(String(100), nullable=False)
    class_name      = Column(String(50), nullable=True)
    parent_phone    = Column(String(20), nullable=True)
    parent_email    = Column(String(255), nullable=True)
    is_enrolled     = Column(Boolean, default=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, server_default=func.now())
    updated_at      = Column(DateTime, onupdate=func.now())

    def __repr__(self):
        return f"<Student {self.first_name} {self.last_name}>"


class FaceEmbedding(Base):
    """
    Stores face embeddings for enrolled students.
    Raw face images are NOT stored — only embeddings.
    This is privacy by design.
    """
    __tablename__ = 'face_embeddings'

    id          = Column(String, primary_key=True, default=generate_uuid)
    student_id  = Column(String, ForeignKey('students.id'), nullable=False)
    embedding   = Column(Text, nullable=False)
    model_name  = Column(String(100), default='buffalo_l')
    created_at  = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<FaceEmbedding student={self.student_id}>"


class AttendanceRecord(Base):
    """
    Records every attendance event.
    Each entry is immutable — corrections
    create new entries with reference to original.
    """
    __tablename__ = 'attendance_records'

    id              = Column(String, primary_key=True, default=generate_uuid)
    student_id      = Column(String, ForeignKey('students.id'), nullable=False)
    school_id       = Column(String, ForeignKey('schools.id'), nullable=False)
    date            = Column(String(10), nullable=False)
    time_recorded   = Column(DateTime, server_default=func.now())
    status          = Column(String(20), default='present')
    confidence      = Column(Float, nullable=True)
    method          = Column(String(20), default='face_recognition')
    is_corrected    = Column(Boolean, default=False)
    correction_note = Column(Text, nullable=True)
    corrected_by    = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<Attendance student={self.student_id} date={self.date}>"


class AuditLog(Base):
    """
    Immutable audit trail of all system actions.
    Every correction, enrollment, and override is logged here.
    This protects the school and Apex Intelligence legally.
    """
    __tablename__ = 'audit_logs'

    id          = Column(String, primary_key=True, default=generate_uuid)
    action      = Column(String(100), nullable=False)
    performed_by= Column(String(255), nullable=True)
    target_id   = Column(String, nullable=True)
    details     = Column(Text, nullable=True)
    timestamp   = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<AuditLog {self.action} at {self.timestamp}>"


class User(Base):
    """
    System users — administrators and teachers.
    """
    __tablename__ = 'users'

    id          = Column(String, primary_key=True, default=generate_uuid)
    school_id   = Column(String, ForeignKey('schools.id'), nullable=False)
    email       = Column(String(255), unique=True, nullable=False)
    full_name   = Column(String(255), nullable=False)
    role        = Column(String(20), default='teacher')
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<User {self.full_name} role={self.role}>"