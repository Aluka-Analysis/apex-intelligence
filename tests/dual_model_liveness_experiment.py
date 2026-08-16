"""
Almanac AI
Dual-Model Passive Liveness Experiment

Tests:
    - MiniFASNetV1SE
    - MiniFASNetV2

The experiment records the 3-class probability output from
both models so we can determine which class corresponds to
REAL and which corresponds to SPOOF under our camera conditions.

IMPORTANT:
Do not change class mappings in production code until this
experiment is completed.
"""

import cv2
import numpy as np
import onnxruntime as ort

from ai.recognition.detector import FaceDetector


V1SE_MODEL = "models/antispoof/MiniFASNetV1SE.onnx"
V2_MODEL = "models/antispoof/MiniFASNetV2.onnx"


class MiniFASNet:

    def __init__(self, model_path, scale):

        self.model_path = model_path
        self.scale = scale

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_size = (80, 80)

    def crop_face(self, image, bbox):

        x1, y1, x2, y2 = [
            int(v) for v in bbox
        ]

        h, w = image.shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            return None

        scale = min(
            (h - 1) / box_h,
            (w - 1) / box_w,
            self.scale
        )

        new_w = box_w * scale
        new_h = box_h * scale

        center_x = x1 + box_w / 2
        center_y = y1 + box_h / 2

        nx1 = max(
            0,
            int(center_x - new_w / 2)
        )

        ny1 = max(
            0,
            int(center_y - new_h / 2)
        )

        nx2 = min(
            w - 1,
            int(center_x + new_w / 2)
        )

        ny2 = min(
            h - 1,
            int(center_y + new_h / 2)
        )

        crop = image[
            ny1:ny2 + 1,
            nx1:nx2 + 1
        ]

        if crop.size == 0:
            return None

        crop = cv2.resize(
            crop,
            self.input_size
        )

        return crop

    def predict(self, face_crop):

        image = face_crop.astype(
            np.float32
        )

        # The yakhyo inference implementation
        # uses raw 0-255 float input.
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: image
            }
        )[0]

        logits = output[0]

        logits = logits - np.max(logits)

        probabilities = np.exp(logits)

        probabilities /= probabilities.sum()

        return probabilities


def main():

    print("=" * 75)
    print("ALMANAC AI — DUAL MODEL PASSIVE LIVENESS EXPERIMENT")
    print("=" * 75)

    print("\nLoading face detector...")

    detector = FaceDetector()

    print("Loading MiniFASNetV1SE...")
    v1se = MiniFASNet(
        V1SE_MODEL,
        scale=4.0
    )

    print("Loading MiniFASNetV2...")
    v2 = MiniFASNet(
        V2_MODEL,
        scale=2.7
    )

    print("\nModels loaded successfully.")

    print("\nIMPORTANT EXPERIMENT PROTOCOL")
    print("-" * 75)
    print("Phase 1: Look directly at the camera.")
    print("         This is the REAL condition.")
    print("")
    print("Phase 2: Hold a clear phone photograph in front")
    print("         of your face.")
    print("         This is the SPOOF condition.")
    print("")
    print("Phase 3: Hold a printed photograph if available.")
    print("         This is another SPOOF condition.")
    print("")
    print("Phase 4: Replay a video of your face if available.")
    print("         This is another SPOOF condition.")
    print("")
    print("Press Q to finish.")
    print("=" * 75)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("\nERROR: Could not open webcam.")

        return

    observations = []

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read frame.")

            break

        frame = cv2.flip(
            frame,
            1
        )

        face = detector.detect_largest_face(
            frame
        )

        if face is not None:

            crop = v2.crop_face(
                frame,
                face.bbox
            )

            if crop is not None:

                p_v1se = v1se.predict(
                    v1se.crop_face(
                        frame,
                        face.bbox
                    )
                )

                p_v2 = v2.predict(
                    crop
                )

                observations.append(
                    (
                        p_v1se,
                        p_v2
                    )
                )

                x1, y1, x2, y2 = [
                    int(v)
                    for v in face.bbox
                ]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                text1 = (
                    f"V1SE: "
                    f"{p_v1se[0]:.2f} "
                    f"{p_v1se[1]:.2f} "
                    f"{p_v1se[2]:.2f}"
                )

                text2 = (
                    f"V2:   "
                    f"{p_v2[0]:.2f} "
                    f"{p_v2[1]:.2f} "
                    f"{p_v2[2]:.2f}"
                )

                cv2.putText(
                    frame,
                    text1,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    text2,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "Almanac AI - Dual Liveness",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()

    if not observations:

        print("\nNo observations collected.")

        return

    v1se_scores = np.array([
        x[0]
        for x in observations
    ])

    v2_scores = np.array([
        x[1]
        for x in observations
    ])

    print("\n")
    print("=" * 75)
    print("EXPERIMENT RESULTS")
    print("=" * 75)

    print("\nMiniFASNetV1SE")
    print("-" * 75)

    for i in range(3):

        print(
            f"Class {i}: "
            f"mean={v1se_scores[:, i].mean():.4f} "
            f"std={v1se_scores[:, i].std():.4f} "
            f"min={v1se_scores[:, i].min():.4f} "
            f"max={v1se_scores[:, i].max():.4f}"
        )

    print("\nMiniFASNetV2")
    print("-" * 75)

    for i in range(3):

        print(
            f"Class {i}: "
            f"mean={v2_scores[:, i].mean():.4f} "
            f"std={v2_scores[:, i].std():.4f} "
            f"min={v2_scores[:, i].min():.4f} "
            f"max={v2_scores[:, i].max():.4f}"
        )

    print("\nFrames analysed:", len(observations))

    print("=" * 75)


if __name__ == "__main__":
    main()