"""
Almanac AI — Liveness Laboratory

Purpose:
    Inspect MiniFASNetV2 behaviour before integrating
    liveness decisions into attendance.

This is a LABORATORY tool.

It does NOT:
    - log attendance
    - decide LIVE/SPOOF
    - apply a production threshold

It only records the model's behaviour.

Test conditions:
    1. Live face
    2. Phone photograph
    3. Printed photograph
    4. Video replay
"""

import cv2
import numpy as np
from statistics import mean

from ai.liveness.passive_liveness import PassiveLivenessDetector
from ai.recognition.detector import FaceDetector


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLES_PER_TEST = 30
CAMERA_INDEX = 0


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def print_result(samples):
    """
    Display statistics from collected samples.
    """

    if not samples:
        print("No samples collected.")
        return

    class_0 = [x[0] for x in samples]
    class_1 = [x[1] for x in samples]
    class_2 = [x[2] for x in samples]

    print("\nRESULTS")
    print("-" * 65)

    for index, values in enumerate(
        [class_0, class_1, class_2]
    ):
        print(
            f"Class {index}: "
            f"Mean={mean(values):.4f}  "
            f"Min={min(values):.4f}  "
            f"Max={max(values):.4f}"
        )

    print("-" * 65)

    highest_classes = [
        int(np.argmax(sample))
        for sample in samples
    ]

    counts = {}

    for cls in highest_classes:
        counts[cls] = counts.get(cls, 0) + 1

    print("Highest-scoring class frequency:")

    for cls, count in sorted(counts.items()):
        percentage = count / len(samples) * 100

        print(
            f"  Class {cls}: "
            f"{count}/{len(samples)} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# MAIN TEST
# ============================================================

def run_test(detector, liveness, test_name):

    print_header(test_name)

    print(
        f"""
Collecting {SAMPLES_PER_TEST} samples.

Position the subject in front of the webcam.

Press SPACE to begin collecting samples.
Press Q to cancel.
"""
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return []

    started = False
    samples = []

    try:

        while len(samples) < SAMPLES_PER_TEST:

            ret, frame = camera.read()

            if not ret:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            display = frame.copy()

            faces = detector.detect(frame)

            if faces:

                # For laboratory testing we use
                # the largest detected face.

                face = max(
                    faces,
                    key=lambda f:
                    (f.bbox[2] - f.bbox[0])
                    * (f.bbox[3] - f.bbox[1])
                )

                x1, y1, x2, y2 = [
                    int(x) for x in face.bbox
                ]

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                if started:

                    face_crop = (
                        liveness.crop_face(
                            frame,
                            face.bbox
                        )
                    )

                    if face_crop is not None:

                        result = (
                            liveness.predict_with_probabilities(
                                face_crop
                            )
                        )

                        probabilities = (
                            result["probabilities"]
                        )

                        raw_scores = (
                            result["raw_scores"]
                        )

                        samples.append(
                            probabilities
                        )

                        print(
                            f"Sample "
                            f"{len(samples):02d}/"
                            f"{SAMPLES_PER_TEST} | "
                            f"Raw: "
                            f"{[round(x, 3) for x in raw_scores]} | "
                            f"Prob: "
                            f"{[round(x, 3) for x in probabilities]} | "
                            f"Highest: "
                            f"{int(np.argmax(probabilities))}"
                        )

                        cv2.putText(
                            display,
                            f"Sample: "
                            f"{len(samples)}/"
                            f"{SAMPLES_PER_TEST}",
                            (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                    else:

                        cv2.putText(
                            display,
                            "Face crop failed",
                            (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                else:

                    cv2.putText(
                        display,
                        "Press SPACE to begin",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

            else:

                cv2.putText(
                    display,
                    "No face detected",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                display,
                "SPACE = start | Q = quit",
                (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Almanac AI - Liveness Laboratory",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") and not started:
                started = True
                print("\nCollecting samples...\n")

    finally:

        camera.release()
        cv2.destroyAllWindows()

    print_result(samples)

    return samples


# ============================================================
# PROGRAM ENTRY
# ============================================================

def main():

    print_header(
        "ALMANAC AI — LIVENESS LABORATORY"
    )

    print(
        """
This experiment does NOT make attendance decisions.

We are measuring MiniFASNetV2 behaviour.

DO NOT change:
    - liveness threshold
    - class mapping
    - attendance engine

yet.

We first collect evidence.
"""
    )

    print("Initializing face detector...")

    detector = FaceDetector()

    print("Initializing liveness detector...")

    liveness = PassiveLivenessDetector()

    print("\nAll laboratory components ready.")

    print(
        """
IMPORTANT:

For each experiment, keep the test condition
consistent throughout the sample collection.

Recommended order:

1. Live face
2. Phone photograph
3. Printed photograph
4. Video replay

Press ENTER to begin.
"""
    )

    input()

    # --------------------------------------------------------
    # RUN 1
    # --------------------------------------------------------

    live_samples = run_test(
        detector,
        liveness,
        "RUN 1 — LIVE FACE"
    )

    # --------------------------------------------------------
    # RUN 2
    # --------------------------------------------------------

    input(
        "\nPress ENTER when ready for "
        "RUN 2 — PHONE PHOTOGRAPH..."
    )

    phone_samples = run_test(
        detector,
        liveness,
        "RUN 2 — PHONE PHOTOGRAPH"
    )

    # --------------------------------------------------------
    # RUN 3
    # --------------------------------------------------------

    input(
        "\nPress ENTER when ready for "
        "RUN 3 — PRINTED PHOTOGRAPH..."
    )

    printed_samples = run_test(
        detector,
        liveness,
        "RUN 3 — PRINTED PHOTOGRAPH"
    )

    # --------------------------------------------------------
    # RUN 4
    # --------------------------------------------------------

    input(
        "\nPress ENTER when ready for "
        "RUN 4 — VIDEO REPLAY..."
    )

    video_samples = run_test(
        detector,
        liveness,
        "RUN 4 — VIDEO REPLAY"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_header(
        "FINAL LABORATORY SUMMARY"
    )

    tests = {
        "Live face": live_samples,
        "Phone photograph": phone_samples,
        "Printed photograph": printed_samples,
        "Video replay": video_samples,
    }

    for name, samples in tests.items():

        if not samples:
            print(
                f"{name:22} → No samples"
            )
            continue

        averages = [
            mean(
                sample[index]
                for sample in samples
            )
            for index in range(3)
        ]

        dominant = int(
            np.argmax(averages)
        )

        print(
            f"{name:22} → "
            f"Class 0={averages[0]:.3f}, "
            f"Class 1={averages[1]:.3f}, "
            f"Class 2={averages[2]:.3f}, "
            f"Dominant={dominant}"
        )

    print(
        """
==============================================================

LAB COMPLETE.

Do NOT interpret Class 0, 1, or 2 yet.

Send me the output from all four runs.

We will use the measurements to determine:

    • class behaviour
    • live/spoof separation
    • score stability
    • webcam suitability
    • appropriate temporal smoothing
    • threshold calibration

Only after that will we reconnect liveness
to the attendance decision.

==============================================================
"""
    )


if __name__ == "__main__":
    main()