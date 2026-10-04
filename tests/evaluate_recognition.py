"""
Almanac AI — Multi-Person Recognition Evaluation

Tests the complete recognition pipeline with multiple
enrolled students to measure:

1. True Positive Rate (TPR)
   → Correct person identified as themselves

2. False Positive Rate (FPR)
   → Wrong person identified as someone else

3. False Negative Rate (FNR)
   → Enrolled person not recognised

4. Confidence score distributions
   → Genuine match scores vs impostor scores

5. Liveness success rate
   → Percentage passing blink check first attempt

This evaluation determines whether the system is
ready for a real school pilot.
"""

import sys
import os
import cv2
import json
import numpy as np
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_local_db, LocalSession
from ai.recognition.detector import FaceDetector
from ai.recognition.embedder import FaceEmbedder
from ai.recognition.matcher import FaceMatcher
from ai.recognition.liveness import LivenessDetector
from ai.enrollment.enroller import StudentEnroller


class RecognitionEvaluator:
    """
    Runs structured evaluation experiments
    and generates an honest performance report.
    """

    def __init__(self):
        init_local_db()
        self.db       = LocalSession()
        enroller      = StudentEnroller()
        self.enrolled = enroller.load_all_embeddings(self.db)

        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.matcher  = FaceMatcher(threshold=0.45)
        self.liveness = LivenessDetector(
            ear_threshold   = 0.25,
            blinks_required = 1,
            session_seconds = 8.0
        )

        # Results storage
        self.results = {
            'genuine_tests':   [],
            'impostor_tests':  [],
            'liveness_tests':  [],
            'metadata': {
                'date':             datetime.now().isoformat(),
                'enrolled_count':   len(self.enrolled),
                'threshold':        0.45,
                'enrolled_students': [
                    s['student_name'] for s in self.enrolled
                ]
            }
        }

    def run_genuine_test(self, expected_name: str, test_num: int) -> dict:
        """
        Test: Correct person stands in front of camera.
        Expected: System identifies them correctly.

        Args:
            expected_name: who is standing in front of camera
            test_num:      test number for labelling

        Returns:
            Test result dict
        """
        print(f"\n--- Genuine Test {test_num}: {expected_name} ---")
        print(f"Ask {expected_name} to stand in front of camera.")
        print("Press SPACE when face is clearly visible. Q to skip.")

        cap = cv2.VideoCapture(0)
        result = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                display = frame.copy()

                cv2.putText(
                    display,
                    f"Genuine Test {test_num}: {expected_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )
                cv2.putText(
                    display,
                    "SPACE to capture — Q to skip",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1
                )

                face = self.detector.detect_largest_face(frame)
                if face is not None:
                    display = self.detector.draw_detection(
                        display, face,
                        label = "Ready",
                        color = (0, 255, 0)
                    )

                cv2.imshow("Almanac AI — Evaluation", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print(f"Skipped test {test_num}")
                    break

                if key == ord(' ') and face is not None:
                    embedding    = self.embedder.extract(face)
                    match_result = self.matcher.match(embedding, self.enrolled)

                    predicted_name = match_result.get('student_name', 'Unknown')
                    confidence     = match_result.get('confidence', 0.0)
                    matched        = match_result['matched']

                    # Check if correct person identified
                    correct = (
                        matched and
                        expected_name.lower() in predicted_name.lower()
                    )

                    result = {
                        'test_num':      test_num,
                        'test_type':     'genuine',
                        'expected':      expected_name,
                        'predicted':     predicted_name if matched else 'Unknown',
                        'confidence':    round(confidence, 4),
                        'matched':       matched,
                        'correct':       correct,
                        'timestamp':     datetime.now().isoformat()
                    }

                    status = "CORRECT" if correct else "WRONG"
                    print(f"  Expected:  {expected_name}")
                    print(f"  Predicted: {predicted_name}")
                    print(f"  Confidence:{confidence:.1%}")
                    print(f"  Result:    {status}")

                    # Show result on screen
                    color = (0, 255, 0) if correct else (0, 0, 255)
                    display = self.detector.draw_detection(
                        display, face,
                        label      = f"{predicted_name} — {status}",
                        confidence = confidence,
                        color      = color
                    )
                    cv2.imshow("Almanac AI — Evaluation", display)
                    cv2.waitKey(1500)
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return result

    def run_impostor_test(
        self,
        impostor_name: str,
        target_name: str,
        test_num: int
    ) -> dict:
        """
        Test: Wrong person stands in front of camera.
        Expected: System REJECTS or identifies correctly.

        Args:
            impostor_name: who is actually standing there
            target_name:   who they are pretending to be
            test_num:      test number

        Returns:
            Test result dict
        """
        print(f"\n--- Impostor Test {test_num} ---")
        print(f"Ask {impostor_name} to stand in front of camera.")
        print(f"They should NOT be identified as {target_name}.")
        print("Press SPACE when face is clearly visible. Q to skip.")

        cap    = cv2.VideoCapture(0)
        result = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                display = frame.copy()

                cv2.putText(
                    display,
                    f"Impostor Test {test_num}: {impostor_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )
                cv2.putText(
                    display,
                    f"Should NOT match {target_name}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1
                )
                cv2.putText(
                    display,
                    "SPACE to capture — Q to skip",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1
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
                    print(f"Skipped test {test_num}")
                    break

                if key == ord(' ') and face is not None:
                    embedding    = self.embedder.extract(face)
                    match_result = self.matcher.match(embedding, self.enrolled)

                    predicted_name = match_result.get('student_name', 'Unknown')
                    confidence     = match_result.get('confidence', 0.0)
                    matched        = match_result['matched']

                    # False accept = system matched someone it should not
                    false_accept = matched

                    # Correct = system rejected the impostor
                    correct = not matched

                    result = {
                        'test_num':     test_num,
                        'test_type':    'impostor',
                        'impostor':     impostor_name,
                        'target':       target_name,
                        'predicted':    predicted_name if matched else 'Unknown',
                        'confidence':   round(confidence, 4),
                        'matched':      matched,
                        'false_accept': false_accept,
                        'correct':      correct,
                        'timestamp':    datetime.now().isoformat()
                    }

                    if false_accept:
                        print(f"  FALSE ACCEPT: {impostor_name} matched as {predicted_name}")
                        print(f"  Confidence: {confidence:.1%}")
                        print(f"  Result: SECURITY FAILURE")
                    else:
                        print(f"  CORRECT REJECTION: {impostor_name} not matched")
                        print(f"  Best score: {confidence:.1%} (below threshold)")
                        print(f"  Result: SECURE")

                    color = (0, 0, 255) if false_accept else (0, 255, 0)
                    label = f"FALSE ACCEPT: {predicted_name}" if false_accept else "CORRECTLY REJECTED"
                    display = self.detector.draw_detection(
                        display, face,
                        label = label,
                        color = color
                    )
                    cv2.imshow("Almanac AI — Evaluation", display)
                    cv2.waitKey(1500)
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return result

    def run_liveness_test(self, person_name: str, test_num: int) -> dict:
        """
        Test liveness detection speed and reliability.
        Measures time from face detection to blink confirmation.

        Args:
            person_name: who is being tested
            test_num:    test number

        Returns:
            Test result with timing data
        """
        print(f"\n--- Liveness Test {test_num}: {person_name} ---")
        print("Stand in front of camera and blink naturally.")
        print("Measuring time to liveness confirmation.")
        print("Press Q to skip.")

        cap              = cv2.VideoCapture(0)
        self.liveness.reset_all()
        result           = None
        start_time       = None
        confirmed_time   = None
        student_id       = 'liveness_test'

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                display = frame.copy()

                cv2.putText(
                    display,
                    f"Liveness Test {test_num}: {person_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2
                )
                cv2.putText(
                    display,
                    "Blink naturally when ready",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1
                )

                face = self.detector.detect_largest_face(frame)

                if face is not None:
                    if start_time is None:
                        start_time = datetime.now()
                        self.liveness.reset_all()
                        print(f"  Face detected — timer started")

                    live_result = self.liveness.update(face, student_id)

                    color = (0, 165, 255)
                    if live_result['is_live']:
                        color = (0, 255, 0)
                        if confirmed_time is None:
                            confirmed_time = datetime.now()
                            elapsed = (confirmed_time - start_time).total_seconds()
                            print(f"  Liveness confirmed in {elapsed:.2f} seconds")

                    display = self.detector.draw_detection(
                        display, face,
                        label = live_result['status'],
                        color = color
                    )

                    cv2.putText(
                        display,
                        f"EAR: {live_result['ear']:.3f}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1
                    )

                    if live_result['is_live'] and confirmed_time:
                        elapsed = (confirmed_time - start_time).total_seconds()
                        result  = {
                            'test_num':          test_num,
                            'test_type':         'liveness',
                            'person':            person_name,
                            'success':           True,
                            'time_to_confirm':   round(elapsed, 2),
                            'blinks_needed':     live_result['blinks_required'],
                            'timestamp':         datetime.now().isoformat()
                        }
                        cv2.putText(
                            display,
                            f"CONFIRMED in {elapsed:.1f}s",
                            (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2
                        )
                        cv2.imshow("Almanac AI — Evaluation", display)
                        cv2.waitKey(2000)
                        break

                cv2.imshow("Almanac AI — Evaluation", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Skipped liveness test")
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

        return result

    def generate_report(self):
        """
        Generate honest evaluation report from all test results.
        """
        genuine   = self.results['genuine_tests']
        impostor  = self.results['impostor_tests']
        liveness  = self.results['liveness_tests']

        print("\n")
        print("=" * 65)
        print("ALMANAC AI — RECOGNITION EVALUATION REPORT")
        print("=" * 65)
        print(f"Date:     {datetime.now().strftime('%A %d %B %Y %H:%M')}")
        print(f"Students: {self.results['metadata']['enrolled_count']}")
        print(f"Threshold:{self.results['metadata']['threshold']}")
        print()

        # Genuine tests
        print("── GENUINE MATCH TESTS ──────────────────────────────────")
        if genuine:
            correct    = sum(1 for r in genuine if r and r.get('correct'))
            total      = len([r for r in genuine if r])
            tpr        = correct / total if total > 0 else 0
            scores     = [r['confidence'] for r in genuine if r and r.get('matched')]
            avg_score  = np.mean(scores) if scores else 0

            print(f"Total tests:          {total}")
            print(f"Correctly identified: {correct}")
            print(f"True Positive Rate:   {tpr:.1%}")
            print(f"Average confidence:   {avg_score:.1%}")

            print("\nDetailed results:")
            for r in genuine:
                if r:
                    status = "CORRECT" if r['correct'] else "WRONG"
                    print(
                        f"  [{status}] Expected: {r['expected']:<20} "
                        f"Got: {r['predicted']:<20} "
                        f"Score: {r['confidence']:.1%}"
                    )
        else:
            print("No genuine tests recorded.")

        print()
        print("── IMPOSTOR REJECTION TESTS ─────────────────────────────")
        if impostor:
            false_accepts  = sum(1 for r in impostor if r and r.get('false_accept'))
            total          = len([r for r in impostor if r])
            far            = false_accepts / total if total > 0 else 0
            rejection_rate = 1 - far

            print(f"Total tests:          {total}")
            print(f"Correctly rejected:   {total - false_accepts}")
            print(f"False accepts:        {false_accepts}")
            print(f"False Accept Rate:    {far:.1%}")
            print(f"Rejection Rate:       {rejection_rate:.1%}")

            print("\nDetailed results:")
            for r in impostor:
                if r:
                    status = "FALSE ACCEPT" if r['false_accept'] else "REJECTED"
                    print(
                        f"  [{status}] Impostor: {r['impostor']:<15} "
                        f"Matched as: {r['predicted']:<20} "
                        f"Score: {r['confidence']:.1%}"
                    )
        else:
            print("No impostor tests recorded.")

        print()
        print("── LIVENESS DETECTION TESTS ─────────────────────────────")
        if liveness:
            successes  = [r for r in liveness if r and r.get('success')]
            total      = len([r for r in liveness if r])
            success_r  = len(successes) / total if total > 0 else 0
            times      = [r['time_to_confirm'] for r in successes if r.get('time_to_confirm')]
            avg_time   = np.mean(times) if times else 0

            print(f"Total tests:          {total}")
            print(f"Successful:           {len(successes)}")
            print(f"Success rate:         {success_r:.1%}")
            print(f"Average time:         {avg_time:.2f} seconds")

            print("\nDetailed results:")
            for r in liveness:
                if r:
                    status = f"{r['time_to_confirm']:.2f}s" if r.get('success') else "FAILED"
                    print(f"  {r['person']:<20} Time: {status}")
        else:
            print("No liveness tests recorded.")

        # Overall assessment
        print()
        print("── OVERALL ASSESSMENT ───────────────────────────────────")

        genuine_total   = len([r for r in genuine if r])
        genuine_correct = sum(1 for r in genuine if r and r['correct'])
        impostor_total  = len([r for r in impostor if r])
        false_accepts   = sum(1 for r in impostor if r and r['false_accept'])

        tpr = genuine_correct / genuine_total if genuine_total > 0 else 0
        far = false_accepts / impostor_total if impostor_total > 0 else 0

        print(f"True Positive Rate:   {tpr:.1%}")
        print(f"False Accept Rate:    {far:.1%}")

        if tpr >= 0.85 and far <= 0.05:
            verdict = "READY FOR PILOT"
            detail  = "Performance meets minimum threshold for school pilot"
        elif tpr >= 0.70 and far <= 0.10:
            verdict = "CONDITIONALLY READY"
            detail  = "Acceptable for supervised pilot with manual override"
        else:
            verdict = "NEEDS IMPROVEMENT"
            detail  = "Threshold tuning or re-enrollment recommended"

        print(f"\nVerdict: {verdict}")
        print(f"Detail:  {detail}")
        print("=" * 65)

        # Save report
        report = {
            'metadata':       self.results['metadata'],
            'genuine_tests':  genuine,
            'impostor_tests': impostor,
            'liveness_tests': liveness,
            'summary': {
                'tpr':     round(tpr, 4),
                'far':     round(far, 4),
                'verdict': verdict
            }
        }

        report_path = f"docs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        os.makedirs('docs', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved: {report_path}")


def run_evaluation():
    """
    Interactive multi-person evaluation session.
    """
    print("=" * 65)
    print("ALMANAC AI — MULTI-PERSON RECOGNITION EVALUATION")
    print("=" * 65)

    evaluator = RecognitionEvaluator()

    if not evaluator.enrolled:
        print("No enrolled students found.")
        print("Run tests/enroll_person.py to enroll students first.")
        return

    print(f"\nEnrolled students ({len(evaluator.enrolled)}):")
    for s in evaluator.enrolled:
        print(f"  → {s['student_name']} — {s['class_name']}")

    print("\n" + "=" * 65)
    print("EVALUATION PLAN")
    print("=" * 65)
    print("""
For each enrolled student:
  Test 1 → Genuine match (correct person)
  Test 2 → Genuine match again (different angle)

For each non-enrolled person available:
  Test 3 → Impostor test (should be rejected)

Liveness tests:
  Test 4 → Measure blink confirmation time

Press ENTER to start. Q during any test to skip it.
""")
    input("Press ENTER to begin evaluation...")

    test_num = 1

    # ── GENUINE TESTS ──
    print("\n" + "=" * 65)
    print("PHASE 1: GENUINE MATCH TESTS")
    print("Each enrolled student tested twice")
    print("=" * 65)

    for student in evaluator.enrolled:
        name = student['student_name']

        for attempt in range(1, 3):
            print(f"\nTest {test_num}: {name} (attempt {attempt}/2)")
            result = evaluator.run_genuine_test(name, test_num)
            if result:
                evaluator.results['genuine_tests'].append(result)
            test_num += 1

    # ── IMPOSTOR TESTS ──
    print("\n" + "=" * 65)
    print("PHASE 2: IMPOSTOR REJECTION TESTS")
    print("Non-enrolled people tested against enrolled database")
    print("=" * 65)

    if len(evaluator.enrolled) >= 2:
        print("\nUsing enrolled students as impostors for each other.")
        for i, student in enumerate(evaluator.enrolled):
            other = evaluator.enrolled[(i + 1) % len(evaluator.enrolled)]
            impostor_name = student['student_name']
            target_name   = other['student_name']

            print(f"\nTest {test_num}: {impostor_name} trying to match as {target_name}")
            result = evaluator.run_impostor_test(impostor_name, target_name, test_num)
            if result:
                evaluator.results['impostor_tests'].append(result)
            test_num += 1

    # Extra impostor test with non-enrolled person
    print(f"\nTest {test_num}: Non-enrolled person test")
    print("Ask someone NOT enrolled to stand in front of camera.")
    non_enrolled = input("Enter their name (or press ENTER to skip): ").strip()

    if non_enrolled:
        result = evaluator.run_impostor_test(
            non_enrolled,
            "any enrolled student",
            test_num
        )
        if result:
            evaluator.results['impostor_tests'].append(result)
        test_num += 1

    # ── LIVENESS TESTS ──
    print("\n" + "=" * 65)
    print("PHASE 3: LIVENESS DETECTION TIMING")
    print("Measuring time to confirm liveness per person")
    print("=" * 65)

    for student in evaluator.enrolled:
        name   = student['student_name']
        print(f"\nTest {test_num}: Liveness timing for {name}")
        result = evaluator.run_liveness_test(name, test_num)
        if result:
            evaluator.results['liveness_tests'].append(result)
        test_num += 1

    # ── GENERATE REPORT ──
    evaluator.generate_report()
    evaluator.db.close()


if __name__ == '__main__':
    run_evaluation()