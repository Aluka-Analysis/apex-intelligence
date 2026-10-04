"""
Pydantic schemas for Almanac AI API.
Defines request and response shapes.
"""

from pydantic import BaseModel
from typing import Optional, List


class StudentCreate(BaseModel):
    first_name:    str
    last_name:     str
    student_id_no: str
    class_name:    str
    parent_phone:  Optional[str] = None
    parent_email:  Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "first_name":    "Chidi",
                "last_name":     "Okafor",
                "student_id_no": "STU001",
                "class_name":    "JSS 2A",
                "parent_phone":  "08098765432"
            }
        }


class StudentResponse(BaseModel):
    id:            str
    student_id_no: str
    first_name:    str
    last_name:     str
    class_name:    str
    parent_phone:  Optional[str] = None
    is_enrolled:   bool
    created_at:    Optional[str] = None


class RecognizeRequest(BaseModel):
    image_base64: str
    session_id:   Optional[str] = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "base64_encoded_jpeg_string",
                "session_id":   "gate_camera_1"
            }
        }


class RecognizeResponse(BaseModel):
    matched:           bool
    student_id:        Optional[str]   = None
    student_name:      Optional[str]   = None
    class_name:        Optional[str]   = None
    confidence:        float
    liveness_passed:   bool
    liveness_status:   str
    attendance_logged: bool
    message:           str


class AttendanceResponse(BaseModel):
    student_id:    str
    first_name:    str
    last_name:     str
    class_name:    str
    status:        str
    time_recorded: Optional[str] = None
    confidence:    Optional[float] = None


class DailyReportResponse(BaseModel):
    date:            str
    total_students:  int
    present:         int
    absent:          int
    attendance_rate: float
    records:         List[AttendanceResponse]


class HealthResponse(BaseModel):
    status:         str
    version:        str
    enrolled_count: int
    models_loaded:  bool
    timestamp:      str


class ModelInfoResponse(BaseModel):
    recognition_model:   str
    embedding_dimension: int
    liveness_method:     str
    threshold:           float
    enrolled_count:      int
    framework:           str


class EnrollmentStatusResponse(BaseModel):
    student_id:      str
    student_name:    str
    captures:        int
    is_enrolled:     bool
    message:         str