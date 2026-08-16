"""
Test passive liveness detection.
Shows real-time anti-spoofing scores on camera feed.

Test this by:
1. Running with your real face — should show REAL
2. Holding a printed photo — should show SPOOF
3. Holding phone showing your photo — should show SPOOF
4. Holding phone playing a video — test and document result
"""

import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.recognition.detector import FaceDetector
from ai.recognition.passive_liveness import PassiveLivenessDetector
from datetime import datetime


def test_passive_liveness():
    """
    Live camera test showing anti-spoofing scores in real time.
    """
    print("=" * 60)
    print("ALMANAC AI — PASSIVE LIVENESS TEST")
    print("Testing: Real Face vs Photo vs Screen vs Video")
    print("=" * 60)
    print("\nInstructions:")
    print("  1. Show your real face — should show REAL (green)")
    print("  2. Hold a printed photo — should show SPOOF (red)")
    print("  3. Show photo on phone screen — should show SPOOF (red)")
    print("  4. Play video of your face — document the result")
    print("  Press Q to quit\n")

    detector = FaceDetector()
    liveness = PassiveLivenessDetector(
        model_path      = 'models/antispoof/minifasnet_v2.onnx',
        real_threshold  = 0.6,
        smoothing_frames = 5
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access camera.")
        return

    frame_count  = 0
    process_every = 2
    results_log  = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.flip(frame, 1)
            display      = frame.copy()
            frame_count += 1

            time_str = datetime.now().strftime('%H:%M:%S')

            if frame_count % process_every == 0:
                face = detector.detect_largest_face(frame)

                if face is not None:
                    result = liveness.check(frame, face, 'test')

                    verdict      = result['verdict']
                    real_score   = result['real_score']
                    smoothed     = result['smoothed_score']
                    frames_seen  = result['frames_seen']

                    # Color by verdict
                    if verdict == 'real':
                        color = (0, 255, 0)
                        label = f"REAL — {smoothed:.1%}"
                    elif verdict == 'spoof':
                        color = (0, 0, 255)
                        label = f"SPOOF — {smoothed:.1%}"
                    else:
                        color = (0, 165, 255)
                        label = f"CHECKING — {smoothed:.1%}"

                    # Draw face box
                    x1, y1, x2, y2 = [int(c) for c in face.bbox]
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                    # Label
                    cv2.putText(
                        display, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2
                    )

                    # Score bars
                    bar_y = y2 + 15
                    bar_w = x2 - x1

                    # Real score bar
                    cv2.putText(
                        display, "Real:",
                        (x1, bar_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1
                    )
                    cv2.rectangle(
                        display,
                        (x1 + 40, bar_y - 10),
                        (x1 + 40 + bar_w - 40, bar_y),
                        (50, 50, 50), -1
                    )
                    real_bar = int((bar_w - 40) * smoothed)
                    if real_bar > 0:
                        cv2.rectangle(
                            display,
                            (x1 + 40, bar_y - 10),
                            (x1 + 40 + real_bar, bar_y),
                            (0, 255, 0), -1
                        )

                    # Spoof score bar
                    bar_y2 = bar_y + 20
                    cv2.putText(
                        display, "Spoof:",
                        (x1, bar_y2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1
                    )
                    cv2.rectangle(
                        display,
                        (x1 + 40, bar_y2 - 10),
                        (x1 + 40 + bar_w - 40, bar_y2),
                        (50, 50, 50), -1
                    )
                    spoof_bar = int((bar_w - 40) * result['spoof_score'])
                    if spoof_bar > 0:
                        cv2.rectangle(
                            display,
                            (x1 + 40, bar_y2 - 10),
                            (x1 + 40 + spoof_bar, bar_y2),
                            (0, 0, 255), -1
                        )

                    # Log result
                    if frame_count % 30 == 0:
                        results_log.append({
                            'time':    time_str,
                            'verdict': verdict,
                            'real':    real_score,
                            'smooth':  smoothed
                        })
                        print(
                            f"[{time_str}] "
                            f"Verdict: {verdict.upper():<9} "
                            f"Real: {real_score:.3f}  "
                            f"Smoothed: {smoothed:.3f}  "
                            f"Frames: {frames_seen}"
                        )

            # HUD
            overlay = display.copy()
            cv2.rectangle(
                overlay, (0, 0),
                (display.shape[1], 70),
                (0, 0, 0), -1
            )
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

            cv2.putText(
                display,
                "ALMANAC AI — Passive Liveness Test",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (220, 220, 220), 1
            )
            cv2.putText(
                display, time_str,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 100), 2
            )
            cv2.putText(
                display, "Q to quit",
                (display.shape[1] - 110, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1
            )

            cv2.imshow("Almanac AI — Passive Liveness", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    if results_log:
        verdicts   = [r['verdict'] for r in results_log]
        real_count = verdicts.count('real')
        spoof_count = verdicts.count('spoof')
        unc_count  = verdicts.count('uncertain')
        total      = len(verdicts)

        print(f"Total readings:  {total}")
        print(f"Real:            {real_count} ({real_count/total:.0%})")
        print(f"Spoof:           {spoof_count} ({spoof_count/total:.0%})")
        print(f"Uncertain:       {unc_count} ({unc_count/total:.0%})")
        avg_real = np.mean([r['real'] for r in results_log])
        print(f"Average real score: {avg_real:.3f}")

    print("=" * 60)


if __name__ == '__main__':
    test_passive_liveness()