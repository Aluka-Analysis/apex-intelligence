"""
Liveness Detection Module.

Prevents presentation attacks using Eye Blink Detection.
Liveness state is tied to student identity not continuous
face presence. A brief loss of face does not reset progress.

Standard dlib 68-point eye indices:
Left eye:  36, 37, 38, 39, 40, 41
Right eye: 42, 43, 44, 45, 46, 47
"""

import numpy as np
import time
from collections import defaultdict


class LivenessDetector:
    """
    Session-based liveness detector.
    Tracks blink progress per student identity
    so brief face loss does not reset progress.
    """

    def __init__(
        self,
        ear_threshold:     float = 0.25,
        blinks_required:   int   = 1,
        session_seconds:   float = 8.0,
        min_frames_closed: int   = 1
    ):
        """
        Initialize session-based liveness detector.

        Args:
            ear_threshold:     EAR below which eye is closed
            blinks_required:   blinks needed to confirm liveness
            session_seconds:   total time per student session
                               Resets only when session expires
                               not when face is briefly lost
            min_frames_closed: minimum frames eye must be closed
        """
        self.ear_threshold     = ear_threshold
        self.blinks_required   = blinks_required
        self.session_seconds   = session_seconds
        self.min_frames_closed = min_frames_closed

        # Per-student session state
        # Key: student_id
        # Value: dict of session data
        self._sessions = {}

        # Current active student
        self.active_student_id = None
        self.last_ear          = 0.3

    def _get_session(self, student_id: str) -> dict:
        """
        Get or create session for a student.
        Sessions persist across brief face losses.
        """
        now = time.time()

        if student_id not in self._sessions:
            self._sessions[student_id] = {
                'blink_count':   0,
                'eye_closed':    False,
                'frames_closed': 0,
                'is_live':       False,
                'start_time':    now,
                'last_seen':     now
            }
        else:
            # Check if session has expired
            session = self._sessions[student_id]
            elapsed = now - session['start_time']

            if elapsed > self.session_seconds and not session['is_live']:
                # Session expired — reset this student
                self._sessions[student_id] = {
                    'blink_count':   0,
                    'eye_closed':    False,
                    'frames_closed': 0,
                    'is_live':       False,
                    'start_time':    now,
                    'last_seen':     now
                }

        self._sessions[student_id]['last_seen'] = now
        return self._sessions[student_id]

    def reset_student(self, student_id: str):
        """Force reset a specific student session."""
        if student_id in self._sessions:
            del self._sessions[student_id]

    def reset_all(self):
        """Reset all sessions."""
        self._sessions = {}
        self.active_student_id = None

    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        try:
            pts = eye_points[:, :2]
            v1  = np.linalg.norm(pts[1] - pts[5])
            v2  = np.linalg.norm(pts[2] - pts[4])
            h   = np.linalg.norm(pts[0] - pts[3])

            if h < 1e-6:
                return 0.3

            return float(np.clip((v1 + v2) / (2.0 * h), 0.0, 1.0))

        except Exception:
            return 0.3

    def get_ear(self, face) -> float:
        """
        Get EAR from InsightFace face object.
        Uses landmark_3d_68 standard 68-point model.
        """
        if (hasattr(face, 'landmark_3d_68') and
                face.landmark_3d_68 is not None and
                len(face.landmark_3d_68) >= 48):

            lm        = face.landmark_3d_68
            left_ear  = self._calculate_ear(lm[36:42])
            right_ear = self._calculate_ear(lm[42:48])
            return (left_ear + right_ear) / 2.0

        if (hasattr(face, 'landmark_2d_106') and
                face.landmark_2d_106 is not None and
                len(face.landmark_2d_106) >= 106):
            try:
                lm = face.landmark_2d_106
                left_eye = np.array([
                    lm[35], lm[41], lm[40],
                    lm[39], lm[38], lm[37]
                ])
                right_eye = np.array([
                    lm[89], lm[95], lm[94],
                    lm[93], lm[92], lm[91]
                ])
                return (self._calculate_ear(left_eye) +
                        self._calculate_ear(right_eye)) / 2.0
            except Exception:
                return 0.3

        return 0.3

    def update(self, face, student_id: str) -> dict:
        """
        Update liveness for a specific student.
        Progress persists across brief face losses.

        Args:
            face:       InsightFace detected face object
            student_id: ID of matched student

        Returns:
            Dict with liveness state
        """
        self.active_student_id = student_id
        session = self._get_session(student_id)
        now     = time.time()
        elapsed = now - session['start_time']

        # Already confirmed live
        if session['is_live']:
            return self._build_result(session, 'Liveness confirmed', False)

        # Get EAR
        ear          = self.get_ear(face)
        self.last_ear = ear

        # Blink state machine
        if ear < self.ear_threshold:
            session['frames_closed'] += 1
            session['eye_closed']     = True
        else:
            if (session['eye_closed'] and
                    session['frames_closed'] >= self.min_frames_closed):
                session['blink_count'] += 1
                print(
                    f"  Blink {session['blink_count']}/{self.blinks_required} "
                    f"detected for {student_id[:8]}  EAR: {ear:.3f}"
                )

            session['eye_closed']    = False
            session['frames_closed'] = 0

        if session['blink_count'] >= self.blinks_required:
            session['is_live'] = True
            return self._build_result(session, 'Liveness confirmed', False)

        remaining = self.blinks_required - session['blink_count']
        time_left = max(0.0, self.session_seconds - elapsed)
        status    = f'Blink {remaining} more time(s)'

        return self._build_result(session, status, False, time_left)

    def is_student_live(self, student_id: str) -> bool:
        """Quick check if student has passed liveness."""
        if student_id not in self._sessions:
            return False
        return self._sessions[student_id]['is_live']

    def _build_result(
        self,
        session: dict,
        status: str,
        timed_out: bool,
        time_left: float = 0.0
    ) -> dict:
        return {
            'is_live':         session['is_live'],
            'blink_count':     session['blink_count'],
            'blinks_required': self.blinks_required,
            'ear':             round(self.last_ear, 3),
            'status':          status,
            'time_left':       round(time_left, 1),
            'timed_out':       timed_out
        }