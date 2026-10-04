"""
Almanac AI — Multi-Person Recognition Evaluation v2
Fixed version with correct impostor logic and camera stability.

Fixes from v1:
1. Impostor logic corrected — enrolled person matching themselves
   is NOT a false accept
2. Camera warmup delay between tests prevents hardware errors
3. Liveness test uses unique student IDs per person
4. Camera opens once per test not repeatedly
"""

import sys
import os
import cv2
import json
import time
import numpy as np
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_local_db, LocalSession
from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder
from ai.recognition.matcher import FaceMatcher
from ai.recognition.liveness import LivenessDetector
from ai.enrollment.enroller import StudentEnroller


def warmup_camera(delay: float = 2.0):
    """
    Allow camera hardware to fully release between tests.
    Prevents cap_msmf errors on Windows.
    """
    print(f"  Camera releasing — please wait {delay:.0f} seconds...")
    time.sleep(delay)


class RecognitionEvaluator:
    """
    Structured evaluation of Almanac AI recognition pipeline.
    Measures TPR, FAR, and liveness performance honestly.
    """

    def __init__(self):
        init_local_db()
        self.db   = LocalSession()

        enroller      = StudentEnroller()
        self.enrolled = enroller.load_all_embeddings(self.db)

        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.matcher  = FaceMatcher(threshold=0.45)
        self.liveness = LivenessDetector(
            ear_threshold   = 0.25,
            blinks_required = 1,
            session_seconds = 10.0
        )

        self.results = {
            'genuine_tests':  [],
            'impostor_tests': [],
            'liveness_tests': [],
            'metadata': {
                'date':              datetime.now().isoformat(),
                'enrolled_count':    len(self.enrolled),
                'threshold':         0.45,
                'enrolled_students': [s['student_name'] for s in self.enrolled]
            }
        }

    def _open_camera(self):
        """Open camera with warmup frames to stabilise exposure."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None

        # Read 10 warmup frames
        for _ in range(10):
            cap.read()

        return cap

    def run_genuine_test(self, expected_name: str, test_num: int) -> dict:
        """
        Test: Correct enrolled person stands in front of camera.
        Expected: System identifies them correctly.
        """
        print(f"\n--- Genuine Test {test_num}: {expected_name} ---")
        print(f"Ask {expected_name} to stand in front of camera.")
        print("Press SPACE to capture — Q to skip.")

        cap    = self._open_camera()
        result = None

        if cap is None:
            print("Camera not available. Skipping.")
            return None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed. Skipping test.")
                    break

                frame   = cv2.flip(frame, 1)
                display = frame.copy()

                # Instructions overlay
                cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
                cv2.putText(
                    display,
                    f"Genuine Test {test_num}: {expected_name}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, (0, 255, 255), 2
                )
                cv2.putText(
                    display,
                    "SPACE to capture — Q to skip",
                    (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (200, 200, 200), 1
                )

                face = self.detector.detect_largest_face(frame)
                if face is not None:
                    display = self.detector.draw_detection(
                        display, face,
                        label = "Ready — press SPACE",
                        color = (0, 255, 0)
                    )

                cv2.imshow("Almanac AI — Evaluation", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print(f"  Skipped test {test_num}")
                    break

                if key == ord(' ') and face is not None:
                    embedding    = self.embedder.extract(face)
                    match_result = self.matcher.match(embedding, self.enrolled)

                    predicted    = match_result.get('student_name', 'Unknown')
                    confidence   = match_result.get('confidence', 0.0)
                    matched      = match_result['matched']

                    # Correct if matched AND predicted name contains expected name
                    correct = (
                        matched and
                        expected_name.lower().split()[0] in predicted.lower()
                    )

                    result = {
                        'test_num':   test_num,
                        'test_type':  'genuine',
                        'expected':   expected_name,
                        'predicted':  predicted if matched else 'Unknown',
                        'confidence': round(confidence, 4),
                        'matched':    matched,
                        'correct':    correct,
                        'timestamp':  datetime.now().isoformat()
                    }

                    status = "CORRECT" if correct else "WRONG"
                    color  = (0, 255, 0) if correct else (0, 0, 255)

                    print(f"  Expected:   {expected_name}")
                    print(f"  Predicted:  {predicted}")
                    print(f"  Confidence: {confidence:.1%}")
                    print(f"  Result:     {status}")

                    display = self.detector.draw_detection(
                        display, face,
                        label      = f"{predicted} — {status}",
                        confidence = confidence,
                        color      = color
                    )
                    cv2.imshow("Almanac AI — Evaluation", display)
                    cv2.waitKey(1500)
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            warmup_camera(1.5)

        return result

    def run_impostor_test(
        self,
        impostor_name: str,
        test_num: int,
        enrolled: bool = False
    ) -> dict:
        """
        Test: A person stands in front of camera.
        Two scenarios:
        A) Enrolled person → should match ONLY as themselves (not as someone else)
        B) Non-enrolled person → should be REJECTED entirely

        Args:
            impostor_name: name of person being tested
            test_num:      test number
            enrolled:      True if this person IS enrolled in the system
        """
        print(f"\n--- Impostor Test {test_num}: {impostor_name} ---")

        if enrolled:
            print(f"Enrolled person cross-test.")
            print(f"{impostor_name} should match as themselves — not as anyone else.")
        else:
            print(f"Non-enrolled person test.")
            print(f"{impostor_name} should be REJECTED entirely.")

        print("Press SPACE to capture — Q to skip.")

        cap    = self._open_camera()
        result = None

        if cap is None:
            print("Camera not available. Skipping.")
            return None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed. Skipping.")
                    break

                frame   = cv2.flip(frame, 1)
                display = frame.copy()

                # Instructions overlay
                cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
                cv2.putText(
                    display,
                    f"Impostor Test {test_num}: {impostor_name}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, (0, 0, 255), 2
                )
                label_2 = (
                    "Should match as THEMSELVES only"
                    if enrolled else
                    "Should be REJECTED"
                )
                cv2.putText(
                    display, label_2,
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (200, 200, 200), 1
                )

                face = self.detector.detect_largest_face(frame)
                if face is not None:
                    display = self.detector.draw_detection(
                        display, face,
                        label = impostor_name,
                        color = (0, 0, 255)
                    )

                cv2.imshow("Almanac AI — Evaluation", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print(f"  Skipped test {test_num}")
                    break

                if key == ord(' ') and face is not None:
                    embedding    = self.embedder.extract(face)
                    match_result = self.matcher.match(embedding, self.enrolled)

                    predicted  = match_result.get('student_name', 'Unknown')
                    confidence = match_result.get('confidence', 0.0)
                    matched    = match_result['matched']

                    if enrolled:
                        # Enrolled person scenario
                        # Correct: matched as THEMSELVES
                        # False accept: matched as SOMEONE ELSE
                        # False reject: not matched at all (also a problem)

                        matched_as_themselves = (
                            matched and
                            impostor_name.lower().split()[0] in predicted.lower()
                        )
                        matched_as_wrong_person = (
                            matched and
                            impostor_name.lower().split()[0] not in predicted.lower()
                        )

                        false_accept = matched_as_wrong_person
                        correct      = matched_as_themselves

                        if matched_as_themselves:
                            verdict = "CORRECT — matched as themselves"
                            color   = (0, 255, 0)
                        elif matched_as_wrong_person:
                            verdict = f"FALSE ACCEPT — matched as {predicted}"
                            color   = (0, 0, 255)
                        else:
                            verdict = "FALSE REJECT — not matched (should have matched)"
                            color   = (0, 165, 255)

                    else:
                        # Non-enrolled person scenario
                        # Correct: rejected (not matched)
                        # False accept: matched as anyone

                        false_accept = matched
                        correct      = not matched

                        if not matched:
                            verdict = f"CORRECTLY REJECTED — score {confidence:.1%}"
                            color   = (0, 255, 0)
                        else:
                            verdict = f"FALSE ACCEPT — matched as {predicted}"
                            color   = (0, 0, 255)

                    result = {
                        'test_num':      test_num,
                        'test_type':     'impostor',
                        'impostor_name': impostor_name,
                        'enrolled':      enrolled,
                        'predicted':     predicted if matched else 'Unknown',
                        'confidence':    round(confidence, 4),
                        'matched':       matched,
                        'false_accept':  false_accept,
                        'correct':       correct,
                        'verdict':       verdict,
                        'timestamp':     datetime.now().isoformat()
                    }

                    print(f"  Person:     {impostor_name}")
                    print(f"  Predicted:  {predicted if matched else 'Unknown'}")
                    print(f"  Confidence: {confidence:.1%}")
                    print(f"  Verdict:    {verdict}")

                    display = self.detector.draw_detection(
                        display, face,
                        label = verdict[:40],
                        color = color
                    )
                    cv2.imshow("Almanac AI — Evaluation", display)
                    cv2.waitKey(1800)
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            warmup_camera(1.5)

        return result

    def run_liveness_test(self, person_name: str, test_num: int) -> dict:
        """
        Measure time from face detection to blink confirmation.
        Uses unique session ID per person to avoid state mixing.
        """
        print(f"\n--- Liveness Test {test_num}: {person_name} ---")
        print("Stand in front of camera and blink naturally.")
        print("Timer starts when your face is detected.")
        print("Press Q to skip.")

        # Use unique session ID per person
        session_id = f"liveness_{person_name.replace(' ', '_').lower()}"
        self.liveness.reset_student(session_id)

        cap            = self._open_camera()
        result         = None
        start_time     = None
        confirmed_time = None

        if cap is None:
            print("Camera not available. Skipping.")
            return None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                display = frame.copy()

                now_str = datetime.now().strftime('%H:%M:%S')

                cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
                cv2.putText(
                    display,
                    f"Liveness Test {test_num}: {person_name}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, (0, 165, 255), 2
                )
                cv2.putText(
                    display,
                    "Blink naturally — timer starts when face detected",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (200, 200, 200), 1
                )

                face = self.detector.detect_largest_face(frame)

                if face is not None:
                    if start_time is None:
                        start_time = time.time()
                        print(f"  Face detected — timer started at {now_str}")

                    elapsed     = time.time() - start_time
                    live_result = self.liveness.update(face, session_id)

                    if live_result['is_live'] and confirmed_time is None:
                        confirmed_time = time.time()
                        duration       = confirmed_time - start_time
                        print(f"  Liveness confirmed in {duration:.2f} seconds")

                    color = (0, 255, 0) if live_result['is_live'] else (0, 165, 255)

                    display = self.detector.draw_detection(
                        display, face,
                        label = live_result['status'],
                        color = color
                    )

                    # EAR display
                    cv2.putText(
                        display,
                        f"EAR: {live_result['ear']:.3f}  "
                        f"Blinks: {live_result['blink_count']}/{live_result['blinks_required']}",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1
                    )

                    # Elapsed timer
                    cv2.putText(
                        display,
                        f"Elapsed: {elapsed:.1f}s",
                        (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (255, 255, 100), 2
                    )

                    if live_result['is_live'] and confirmed_time:
                        duration = confirmed_time - start_time
                        result   = {
                            'test_num':        test_num,
                            'test_type':       'liveness',
                            'person':          person_name,
                            'success':         True,
                            'time_to_confirm': round(duration, 2),
                            'ear_at_blink':    live_result['ear'],
                            'timestamp':       datetime.now().isoformat()
                        }

                        cv2.putText(
                            display,
                            f"CONFIRMED in {duration:.1f}s",
                            (10, 165),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2
                        )
                        cv2.imshow("Almanac AI — Evaluation", display)
                        cv2.waitKey(2000)
                        break

                else:
                    if start_time is not None:
                        cv2.putText(
                            display,
                            "Face lost — please reposition",
                            (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.60, (0, 0, 255), 2
                        )

                cv2.imshow("Almanac AI — Evaluation", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print(f"  Skipped liveness test")
                    result = {
                        'test_num':        test_num,
                        'test_type':       'liveness',
                        'person':          person_name,
                        'success':         False,
                        'time_to_confirm': None,
                        'timestamp':       datetime.now().isoformat()
                    }
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            warmup_camera(1.5)

        return result

    def generate_report(self):
        """Generate honest evaluation report."""
        genuine  = [r for r in self.results['genuine_tests']  if r]
        impostor = [r for r in self.results['impostor_tests'] if r]
        liveness = [r for r in self.results['liveness_tests'] if r]

        print("\n")
        print("=" * 65)
        print("ALMANAC AI — RECOGNITION EVALUATION REPORT v2")
        print("=" * 65)
        print(f"Date:      {datetime.now().strftime('%A %d %B %Y %H:%M')}")
        print(f"Students:  {self.results['metadata']['enrolled_count']}")
        print(f"Threshold: {self.results['metadata']['threshold']}")
        print()

        # ── Genuine tests ──
        print("── GENUINE MATCH TESTS ──────────────────────────────────")
        if genuine:
            correct   = sum(1 for r in genuine if r['correct'])
            total     = len(genuine)
            tpr       = correct / total if total > 0 else 0
            scores    = [r['confidence'] for r in genuine if r['matched']]
            avg_score = np.mean(scores) if scores else 0

            print(f"Total tests:          {total}")
            print(f"Correctly identified: {correct}")
            print(f"True Positive Rate:   {tpr:.1%}")
            print(f"Average confidence:   {avg_score:.1%}")
            print("\nDetailed results:")
            for r in genuine:
                status = "CORRECT" if r['correct'] else "WRONG"
                print(
                    f"  [{status:<7}] "
                    f"Expected: {r['expected']:<25} "
                    f"Got: {r['predicted']:<25} "
                    f"Score: {r['confidence']:.1%}"
                )
        else:
            print("No genuine tests recorded.")
            tpr = 0

        print()

        # ── Impostor tests ──
        print("── IMPOSTOR REJECTION TESTS ─────────────────────────────")

        # Separate enrolled cross-tests from non-enrolled tests
        enrolled_cross = [r for r in impostor if r.get('enrolled')]
        non_enrolled   = [r for r in impostor if not r.get('enrolled')]

        if enrolled_cross:
            print("\nCross-identity tests (enrolled students vs each other):")
            false_identity = sum(1 for r in enrolled_cross if r['false_accept'])
            total_cross    = len(enrolled_cross)
            print(f"  Tests:             {total_cross}")
            print(f"  False identity:    {false_identity}")
            print(f"  Correct identity:  {total_cross - false_identity}")

            for r in enrolled_cross:
                print(
                    f"  [{r['verdict'][:35]:<37}] "
                    f"{r['impostor_name']:<20} "
                    f"Score: {r['confidence']:.1%}"
                )

        if non_enrolled:
            print("\nNon-enrolled person rejection tests:")
            false_accepts = sum(1 for r in non_enrolled if r['false_accept'])
            total_ne      = len(non_enrolled)
            far           = false_accepts / total_ne if total_ne > 0 else 0

            print(f"  Tests:             {total_ne}")
            print(f"  Correctly rejected:{total_ne - false_accepts}")
            print(f"  False accepts:     {false_accepts}")
            print(f"  False Accept Rate: {far:.1%}")

            for r in non_enrolled:
                status = "FALSE ACCEPT" if r['false_accept'] else "REJECTED"
                print(
                    f"  [{status:<12}] "
                    f"{r['impostor_name']:<20} "
                    f"Score: {r['confidence']:.1%}"
                )
        else:
            far = 0.0

        print()

        # ── Liveness tests ──
        print("── LIVENESS DETECTION TESTS ─────────────────────────────")
        if liveness:
            successes = [r for r in liveness if r.get('success')]
            total     = len(liveness)
            success_r = len(successes) / total if total > 0 else 0
            times     = [r['time_to_confirm'] for r in successes if r.get('time_to_confirm')]
            avg_time  = np.mean(times) if times else 0
            min_time  = np.min(times)  if times else 0
            max_time  = np.max(times)  if times else 0

            print(f"Total tests:          {total}")
            print(f"Successful:           {len(successes)}")
            print(f"Success rate:         {success_r:.1%}")
            if times:
                print(f"Average time:         {avg_time:.2f} seconds")
                print(f"Fastest:              {min_time:.2f} seconds")
                print(f"Slowest:              {max_time:.2f} seconds")

            print("\nDetailed results:")
            for r in liveness:
                if r.get('success') and r.get('time_to_confirm'):
                    status = f"{r['time_to_confirm']:.2f}s"
                    ear    = f" EAR: {r.get('ear_at_blink', 0):.3f}"
                else:
                    status = "FAILED/SKIPPED"
                    ear    = ""
                print(f"  {r['person']:<25} Time: {status}{ear}")
        else:
            print("No liveness tests recorded.")
            avg_time = 0

        # ── Overall assessment ──
        print()
        print("── OVERALL ASSESSMENT ───────────────────────────────────")

        genuine_total   = len(genuine)
        genuine_correct = sum(1 for r in genuine if r['correct'])
        tpr             = genuine_correct / genuine_total if genuine_total > 0 else 0

        non_enrolled_total = len(non_enrolled)
        false_accepts_ne   = sum(1 for r in non_enrolled if r['false_accept'])
        far                = false_accepts_ne / non_enrolled_total if non_enrolled_total > 0 else 0

        print(f"True Positive Rate:   {tpr:.1%}")
        print(f"False Accept Rate:    {far:.1%}  (non-enrolled persons only)")

        liveness_times = [r['time_to_confirm'] for r in liveness if r.get('success') and r.get('time_to_confirm')]
        avg_liveness   = np.mean(liveness_times) if liveness_times else None

        if avg_liveness:
            print(f"Avg Liveness Time:    {avg_liveness:.2f} seconds")

        print()

        if genuine_total == 0:
            verdict = "INSUFFICIENT DATA"
            detail  = "Run more genuine tests before assessing readiness"
        elif tpr >= 0.85 and far <= 0.05:
            verdict = "READY FOR PILOT"
            detail  = "Performance meets minimum threshold for school pilot"
        elif tpr >= 0.70 and far <= 0.15:
            verdict = "CONDITIONALLY READY"
            detail  = "Acceptable for supervised pilot with manual override in place"
        else:
            verdict = "NEEDS IMPROVEMENT"
            detail  = "Review threshold settings and re-enrollment quality"

        print(f"Verdict:  {verdict}")
        print(f"Detail:   {detail}")

        # Liveness assessment
        if avg_liveness:
            if avg_liveness <= 3.0:
                liveness_verdict = "EXCELLENT — under 3 seconds"
            elif avg_liveness <= 6.0:
                liveness_verdict = "ACCEPTABLE — under 6 seconds"
            else:
                liveness_verdict = "NEEDS CALIBRATION — too slow for school gate"
            print(f"Liveness: {liveness_verdict}")

        print("=" * 65)

        # Save report
        report = {
            'metadata':       self.results['metadata'],
            'genuine_tests':  genuine,
            'impostor_tests': impostor,
            'liveness_tests': liveness,
            'summary': {
                'tpr':             round(tpr, 4),
                'far':             round(far, 4),
                'avg_liveness_s':  round(avg_liveness, 2) if avg_liveness else None,
                'verdict':         verdict
            }
        }

        os.makedirs('docs', exist_ok=True)
        report_path = f"docs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved: {report_path}")


def run_evaluation():
    """Interactive evaluation session."""
    print("=" * 65)
    print("ALMANAC AI — MULTI-PERSON RECOGNITION EVALUATION v2")
    print("=" * 65)

    evaluator = RecognitionEvaluator()

    if not evaluator.enrolled:
        print("No enrolled students found.")
        print("Run tests/enroll_person.py first.")
        return

    print(f"\nEnrolled students ({len(evaluator.enrolled)}):")
    for s in evaluator.enrolled:
        print(f"  → {s['student_name']} — {s['class_name']}")

    print("\n" + "=" * 65)
    print("WHAT THIS EVALUATION MEASURES")
    print("=" * 65)
    print("""
Phase 1 — Genuine Match Tests:
  Each enrolled student tested TWICE
  Expected: System identifies them correctly both times

Phase 2 — Cross-Identity Tests:
  Each enrolled student tested against other enrolled students
  Expected: System identifies them as THEMSELVES
  NOT as a different enrolled student

Phase 3 — Non-Enrolled Rejection Tests:
  Person NOT in database tested
  Expected: System REJECTS them entirely
  This is the most critical security test

Phase 4 — Liveness Timing:
  How fast does blink detection work?
  Target: under 3 seconds

Press ENTER to begin. Q during any test to skip it.
""")
    input("Press ENTER to start evaluation...")

    test_num = 1

    # ── PHASE 1: Genuine tests ──
    print("\n" + "=" * 65)
    print("PHASE 1: GENUINE MATCH TESTS")
    print("=" * 65)

    for student in evaluator.enrolled:
        name = student['student_name']
        for attempt in range(1, 3):
            print(f"\nTest {test_num}: {name} (attempt {attempt}/2)")
            result = evaluator.run_genuine_test(name, test_num)
            if result:
                evaluator.results['genuine_tests'].append(result)
            test_num += 1

    # ── PHASE 2: Cross-identity tests ──
    print("\n" + "=" * 65)
    print("PHASE 2: CROSS-IDENTITY TESTS")
    print("Enrolled students tested — should match as themselves only")
    print("=" * 65)

    for student in evaluator.enrolled:
        name = student['student_name']
        print(f"\nTest {test_num}: {name} — cross identity check")
        result = evaluator.run_impostor_test(name, test_num, enrolled=True)
        if result:
            evaluator.results['impostor_tests'].append(result)
        test_num += 1

    # ── PHASE 3: Non-enrolled rejection tests ──
    print("\n" + "=" * 65)
    print("PHASE 3: NON-ENROLLED PERSON REJECTION")
    print("Critical security test — should all be rejected")
    print("=" * 65)

    print("\nHow many non-enrolled people can you test? (enter 0 to skip)")
    try:
        count = int(input("Number: ").strip())
    except ValueError:
        count = 0

    for i in range(count):
        name = input(f"\nNon-enrolled person {i+1} name: ").strip()
        if name:
            print(f"\nTest {test_num}: {name} — should be rejected")
            result = evaluator.run_impostor_test(name, test_num, enrolled=False)
            if result:
                evaluator.results['impostor_tests'].append(result)
            test_num += 1

    # ── PHASE 4: Liveness timing ──
    print("\n" + "=" * 65)
    print("PHASE 4: LIVENESS DETECTION TIMING")
    print("Measuring seconds to blink confirmation per person")
    print("=" * 65)

    for student in evaluator.enrolled:
        name = student['student_name']
        print(f"\nTest {test_num}: Liveness timing — {name}")
        result = evaluator.run_liveness_test(name, test_num)
        if result:
            evaluator.results['liveness_tests'].append(result)
        test_num += 1

    # ── Generate report ──
    evaluator.generate_report()
    evaluator.db.close()


if __name__ == '__main__':
    run_evaluation()
