"""
Passive Liveness Detection Module.

Uses MiniFASNetV2 — a lightweight anti-spoofing CNN
that silently determines whether a face belongs to
a real live person or a spoof attempt (photo, screen,
video replay).

Key advantage over active liveness (blink/head turn):
→ Student does NOTHING extra
→ Runs in under 50ms per frame
→ No queue created at school gate
→ 5000 students handled at full walking speed

Model: MiniFASNetV2 (ONNX)
Input:  80 x 80 RGB face crop
Output: [spoof_score, real_score, unknown_score]
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
from collections import deque


class PassiveLivenessDetector:
    """
    Silent anti-spoofing detector.
    Analyses face texture and depth cues to distinguish
    real faces from photos, screens, and video replays.
    """

    def __init__(
        self,
        model_path:        str   = 'models/antispoof/minifasnet_v2.onnx',
        real_threshold:    float = 0.6,
        smoothing_frames:  int   = 5
    ):
        """
        Initialize passive liveness detector.

        Args:
            model_path:       path to MiniFASNetV2 ONNX model
            real_threshold:   minimum real score to pass liveness
                              0.6 = confident but not too strict
                              Higher = stricter = fewer false passes

            smoothing_frames: number of frames to average scores over
                              Reduces flickering decisions
                              5 frames at 30fps = 0.17 seconds
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Anti-spoofing model not found: {model_path}\n"
                f"Run the model download command first."
            )

        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        self.input_name    = self.session.get_inputs()[0].name
        self.real_threshold = real_threshold

        # Smoothing buffer per student
        # Key: student_id → deque of recent real scores
        self._score_buffers = {}
        self.smoothing_frames = smoothing_frames

        print(f"Passive liveness model loaded.")
        print(f"Real threshold: {real_threshold:.0%}")

    def _preprocess_face(
        self,
        frame: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """
        Crop and preprocess face region for model input.

        Args:
            frame: full BGR camera frame
            bbox:  face bounding box [x1, y1, x2, y2]

        Returns:
            Preprocessed face array [1, 3, 80, 80]
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]

        # Add padding around face for context
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Crop face
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        # Resize to model input size
        face_resized = cv2.resize(face_crop, (80, 80))

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1
        face_norm = face_rgb.astype(np.float32) / 255.0

        # Transpose to [C, H, W] then add batch dimension
        face_chw  = face_norm.transpose(2, 0, 1)
        face_batch = np.expand_dims(face_chw, axis=0)

        return face_batch

    def check(
        self,
        frame: np.ndarray,
        face,
        student_id: str = 'default'
    ) -> dict:
        """
        Silently check if face is real or spoof.
        Averages scores across recent frames for stability.

        Args:
            frame:      full BGR camera frame
            face:       InsightFace detected face object
            student_id: ID for per-student score buffering

        Returns:
            Dict with liveness result:
            - is_live:      True if real person detected
            - real_score:   confidence it is a real face (0-1)
            - spoof_score:  confidence it is a spoof (0-1)
            - smoothed_score: average across recent frames
            - verdict:      'real', 'spoof', or 'uncertain'
        """
        try:
            face_input = self._preprocess_face(frame, face.bbox)

            if face_input is None:
                return self._uncertain_result()

            # Run inference
            outputs = self.session.run(
                None,
                {self.input_name: face_input}
            )

            scores = outputs[0][0]

            # Apply softmax to get probabilities
            exp_scores  = np.exp(scores - np.max(scores))
            probs       = exp_scores / exp_scores.sum()

            spoof_score = float(probs[0])
            real_score  = float(probs[2])

            # Smooth scores across frames
            if student_id not in self._score_buffers:
                self._score_buffers[student_id] = deque(
                    maxlen=self.smoothing_frames
                )

            self._score_buffers[student_id].append(real_score)
            smoothed = float(np.mean(self._score_buffers[student_id]))

            is_live = smoothed >= self.real_threshold

            if smoothed >= self.real_threshold:
                verdict = 'real'
            elif smoothed < (1 - self.real_threshold):
                verdict = 'spoof'
            else:
                verdict = 'uncertain'

            return {
                'is_live':       is_live,
                'real_score':    round(real_score, 4),
                'spoof_score':   round(spoof_score, 4),
                'smoothed_score': round(smoothed, 4),
                'verdict':       verdict,
                'frames_seen':   len(self._score_buffers[student_id])
            }

        except Exception as e:
            return self._uncertain_result(str(e))

    def reset_student(self, student_id: str):
        """Clear score buffer for a specific student."""
        if student_id in self._score_buffers:
            del self._score_buffers[student_id]

    def reset_all(self):
        """Clear all score buffers."""
        self._score_buffers = {}

    def _uncertain_result(self, error: str = '') -> dict:
        """Return safe default when check cannot be performed."""
        return {
            'is_live':        False,
            'real_score':     0.0,
            'spoof_score':    0.0,
            'smoothed_score': 0.0,
            'verdict':        'uncertain',
            'frames_seen':    0,
            'error':          error
        }