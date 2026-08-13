"""
Passive Face Liveness Module.

Uses MiniFASNetV2 through ONNX Runtime.

This module does NOT immediately decide whether a single
frame is live or spoof.

Instead, it provides:
    - raw model scores
    - probabilities
    - temporal observations
    - LIVE / SPOOF / UNCERTAIN decision

The purpose is to make the liveness component measurable
and configurable before it is trusted for attendance.
"""

from pathlib import Path
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort


class PassiveLivenessDetector:
    """
    MiniFASNetV2 passive face anti-spoofing detector.

    The detector observes multiple consecutive frames before
    producing a liveness decision.
    """

    def __init__(
        self,
        model_path: str = "models/antispoof/minifasnet_v2.onnx",
        min_live_score: float = 0.60,
        min_spoof_score: float = 0.40,
        required_frames: int = 5,
        min_live_frames: int = 3,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Liveness model not found: {self.model_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.min_live_score = min_live_score
        self.min_spoof_score = min_spoof_score
        self.required_frames = required_frames
        self.min_live_frames = min_live_frames

        # History is maintained independently for each identity.
        self.history = {}

        print("Passive liveness model loaded.")
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
        print(f"Live score threshold: {self.min_live_score:.0%}")
        print(f"Spoof score threshold: {self.min_spoof_score:.0%}")
        print(f"Required frames: {self.required_frames}")
        print(f"Minimum live frames: {self.min_live_frames}")

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Prepare face crop for MiniFASNetV2.

        Expected input:
            [batch, 3, 80, 80]
        """

        if face_crop is None or face_crop.size == 0:
            raise ValueError("Invalid face crop.")

        resized = cv2.resize(
            face_crop,
            (80, 80),
            interpolation=cv2.INTER_LINEAR
        )

        # HWC -> CHW
        tensor = resized.transpose(2, 0, 1)

        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)

        return tensor.astype(np.float32)

    # =========================================================
    # SINGLE FRAME INFERENCE
    # =========================================================

    def predict(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Return raw MiniFASNetV2 output.
        """

        input_tensor = self.preprocess(face_crop)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        return outputs[0][0]

    def predict_with_probabilities(
        self,
        face_crop: np.ndarray
    ) -> dict:
        """
        Run inference and convert raw scores to probabilities.
        """

        raw_scores = self.predict(face_crop)

        # Numerically stable softmax
        exp_scores = np.exp(
            raw_scores - np.max(raw_scores)
        )

        probabilities = exp_scores / np.sum(exp_scores)

        return {
            "raw_scores": raw_scores.tolist(),
            "probabilities": probabilities.tolist()
        }

    # =========================================================
    # FACE CROPPING
    # =========================================================

    @staticmethod
    def crop_face(
        frame: np.ndarray,
        bbox
    ) -> np.ndarray | None:
        """
        Crop detected face from OpenCV frame.
        """

        if frame is None or bbox is None:
            return None

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = [
            int(value) for value in bbox
        ]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    # =========================================================
    # TEMPORAL LIVENESS
    # =========================================================

    def observe(
        self,
        face_crop: np.ndarray,
        identity: str = "default"
    ) -> dict:
        """
        Observe one frame and update the temporal liveness state.

        Returns:
            LIVE
            SPOOF
            UNCERTAIN
            CHECKING
        """

        result = self.predict_with_probabilities(face_crop)

        probabilities = result["probabilities"]

        # -----------------------------------------------------
        # IMPORTANT
        # -----------------------------------------------------
        # We are currently using the same class mapping that
        # was used during our experimental calibration.
        #
        # This remains configurable and should be validated
        # further before production deployment.
        # -----------------------------------------------------

        live_score = float(probabilities[1])
        spoof_score = float(probabilities[0])

        if identity not in self.history:
            self.history[identity] = deque(
                maxlen=self.required_frames
            )

        history = self.history[identity]

        history.append({
            "live_score": live_score,
            "spoof_score": spoof_score
        })

        frames_seen = len(history)

        # -----------------------------------------------------
        # Not enough observations yet
        # -----------------------------------------------------

        if frames_seen < self.required_frames:

            return {
                "status": "checking",
                "decision": "UNCERTAIN",
                "live_score": live_score,
                "spoof_score": spoof_score,
                "smoothed_live_score": live_score,
                "frames_seen": frames_seen,
                "required_frames": self.required_frames,
                "live_frames": 0
            }

        # -----------------------------------------------------
        # Calculate temporal averages
        # -----------------------------------------------------

        live_scores = [
            item["live_score"]
            for item in history
        ]

        spoof_scores = [
            item["spoof_score"]
            for item in history
        ]

        average_live = float(
            np.mean(live_scores)
        )

        average_spoof = float(
            np.mean(spoof_scores)
        )

        live_frames = sum(
            score >= self.min_live_score
            for score in live_scores
        )

        spoof_frames = sum(
            score >= (1.0 - self.min_live_score)
            for score in spoof_scores
        )

        # -----------------------------------------------------
        # LIVE
        # -----------------------------------------------------

        if (
            average_live >= self.min_live_score
            and live_frames >= self.min_live_frames
        ):

            decision = "LIVE"

        # -----------------------------------------------------
        # SPOOF
        # -----------------------------------------------------

        elif (
            average_spoof >= self.min_live_score
            and spoof_frames >= self.min_live_frames
        ):

            decision = "SPOOF"

        # -----------------------------------------------------
        # UNCERTAIN
        # -----------------------------------------------------

        else:

            decision = "UNCERTAIN"

        return {
            "status": "decision",
            "decision": decision,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "smoothed_live_score": round(
                average_live,
                4
            ),
            "smoothed_spoof_score": round(
                average_spoof,
                4
            ),
            "frames_seen": frames_seen,
            "required_frames": self.required_frames,
            "live_frames": live_frames,
            "spoof_frames": spoof_frames
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset_identity(self, identity: str):
        """
        Reset temporal history for one identity.
        """

        self.history.pop(identity, None)

    def reset_all(self):
        """
        Reset all temporal histories.
        """

        self.history.clear()