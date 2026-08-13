"""
Almanac AI - Passive Liveness Experiment

Records MiniFASNetV2 output statistics so we can
compare live faces against spoof attempts.
"""

import cv2
import numpy as np

from ai.recognition.detector import FaceDetector
from ai.liveness.passive_liveness import PassiveLivenessDetector


def main():

    print("=" * 65)
    print("ALMANAC AI — PASSIVE LIVENESS EXPERIMENT")
    print("=" * 65)

    detector = FaceDetector()
    liveness = PassiveLivenessDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    scores = []

    print("\nCamera started.")
    print("Collecting scores...")
    print("Press Q to stop the experiment.\n")

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        face = detector.detect_largest_face(frame)

        if face is not None:

            face_crop = liveness.crop_face(
                frame,
                face.bbox
            )

            if face_crop is not None:

                result = liveness.predict_with_probabilities(
                    face_crop
                )

                probabilities = result["probabilities"]

                scores.append(probabilities)

                x1, y1, x2, y2 = [
                    int(value) for value in face.bbox
                ]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                text = (
                    f"{probabilities[0]:.3f} | "
                    f"{probabilities[1]:.3f} | "
                    f"{probabilities[2]:.3f}"
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "Almanac AI - Liveness Experiment",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    # -------------------------------------------------
    # EXPERIMENT SUMMARY
    # -------------------------------------------------

    if not scores:
        print("\nNo face observations collected.")
        return

    scores = np.array(scores)

    print("\n")
    print("=" * 65)
    print("EXPERIMENT RESULTS")
    print("=" * 65)

    for i in range(3):

        print(f"\nCLASS {i}")

        print(f"Minimum : {scores[:, i].min():.4f}")
        print(f"Maximum : {scores[:, i].max():.4f}")
        print(f"Mean    : {scores[:, i].mean():.4f}")
        print(f"Std Dev : {scores[:, i].std():.4f}")

    print("\n")
    print(f"Frames analysed: {len(scores)}")

    print("=" * 65)


if __name__ == "__main__":
    main()