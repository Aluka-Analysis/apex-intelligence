"""
ALMANAC AI — ATTENDANCE ENGINE

Current stage:
    DIAGNOSTIC LIVENESS VALIDATION

Pipeline:

    Camera Frame
         ↓
    Face Detection
         ↓
    Face Quality
         ↓
    Face Embedding
         ↓
    Face Recognition
         ↓
    Face Crop
         ↓
    MiniFASNetV2 Liveness
         ↓
    Class Probability Analysis
         ↓
    Temporal Smoothing
         ↓
    Diagnostic Output

IMPORTANT:

Attendance logging is DISABLED.

This version is specifically designed to determine how
the installed MiniFASNetV2 model behaves with real
webcam input.

We DO NOT assume that:

    class 0 = real
    class 1 = real
    class 2 = real

Instead, all three classes are reported.

The class mapping must be experimentally validated
before liveness is used as a security decision.
"""


class AttendanceEngine:
    """
    Coordinates:

        - face detection
        - face quality checking
        - face embedding
        - face recognition
        - passive liveness
        - temporal smoothing
        - diagnostic reporting

    Attendance database writes are intentionally disabled.
    """

    def __init__(
        self,
        detector,
        embedder,
        matcher,
        liveness_detector,
        real_threshold: float = 0.60,
        min_liveness_frames: int = 5,
        min_live_frames: int = 3
    ):

        self.detector = detector

        self.embedder = embedder

        self.matcher = matcher

        self.liveness = liveness_detector

        # -----------------------------------------------------
        # LIVENESS CONFIGURATION
        # -----------------------------------------------------

        self.real_threshold = (
            float(real_threshold)
        )

        self.min_liveness_frames = (
            int(min_liveness_frames)
        )

        self.min_live_frames = (
            int(min_live_frames)
        )

        # -----------------------------------------------------
        # SESSION STATE
        # -----------------------------------------------------

        self.liveness_history = {}

        self.recognized_students = set()

        self.logged_students = set()

    # =========================================================
    # PROCESS FRAME
    # =========================================================

    def process_frame(
        self,
        frame,
        enrolled_embeddings,
        db=None,
        school_id=None
    ):
        """
        Process one camera frame.

        Returns:

            list[dict]

        Multiple faces can be returned.

        Attendance is NOT written to the database.
        """

        # =====================================================
        # 1. FACE DETECTION
        # =====================================================

        try:

            faces = self.detector.detect(
                frame
            )

        except Exception as error:

            print(
                "\n[DETECTOR ERROR]"
            )

            print(
                str(error)
            )

            return []

        results = []

        # =====================================================
        # NO FACE
        # =====================================================

        if not faces:

            return results

        # =====================================================
        # PROCESS EACH FACE
        # =====================================================

        for face in faces:

            # =================================================
            # 2. FACE QUALITY
            # =================================================

            try:

                quality_ok, reason = (
                    self.detector
                    .is_face_quality_acceptable(
                        face
                    )
                )

            except Exception as error:

                results.append({

                    "status":
                        "quality_error",

                    "error":
                        str(error),

                    "face":
                        face
                })

                continue

            if not quality_ok:

                results.append({

                    "status":
                        "poor_quality",

                    "reason":
                        reason,

                    "face":
                        face
                })

                continue

            # =================================================
            # 3. FACE EMBEDDING
            # =================================================

            try:

                embedding = (
                    self.embedder.extract(
                        face
                    )
                )

            except Exception as error:

                results.append({

                    "status":
                        "embedding_error",

                    "error":
                        str(error),

                    "face":
                        face
                })

                continue

            # =================================================
            # 4. FACE RECOGNITION
            # =================================================

            try:

                match = self.matcher.match(
                    embedding,
                    enrolled_embeddings
                )

            except Exception as error:

                results.append({

                    "status":
                        "recognition_error",

                    "error":
                        str(error),

                    "face":
                        face
                })

                continue

            # =================================================
            # UNKNOWN PERSON
            # =================================================

            if not match.get(
                "matched",
                False
            ):

                results.append({

                    "status":
                        "unknown",

                    "confidence":
                        match.get(
                            "confidence",
                            0.0
                        ),

                    "face":
                        face
                })

                continue

            # =================================================
            # RECOGNIZED STUDENT
            # =================================================

            student_id = match.get(
                "student_id"
            )

            student_name = match.get(
                "student_name",
                "Unknown"
            )

            recognition_confidence = float(
                match.get(
                    "confidence",
                    0.0
                )
            )

            # -------------------------------------------------
            # SESSION TRACKING
            # -------------------------------------------------

            if student_id is not None:

                self.recognized_students.add(
                    student_id
                )

            # =================================================
            # 5. FACE CROP
            # =================================================

            try:

                face_crop = (
                    self.liveness.crop_face(
                        frame,
                        face.bbox
                    )
                )

            except Exception as error:

                results.append({

                    "status":
                        "liveness_error",

                    "student_id":
                        student_id,

                    "student_name":
                        student_name,

                    "confidence":
                        recognition_confidence,

                    "error":
                        str(error),

                    "face":
                        face
                })

                continue

            if face_crop is None:

                results.append({

                    "status":
                        "liveness_error",

                    "student_id":
                        student_id,

                    "student_name":
                        student_name,

                    "confidence":
                        recognition_confidence,

                    "error":
                        "Unable to crop detected face.",

                    "face":
                        face
                })

                continue

            # =================================================
            # 6. PASSIVE LIVENESS
            # =================================================

            try:

                liveness_result = (
                    self.liveness
                    .predict_with_probabilities(
                        face_crop
                    )
                )

            except Exception as error:

                results.append({

                    "status":
                        "liveness_error",

                    "student_id":
                        student_id,

                    "student_name":
                        student_name,

                    "confidence":
                        recognition_confidence,

                    "error":
                        str(error),

                    "face":
                        face
                })

                continue

            # =================================================
            # 7. EXTRACT MODEL OUTPUT
            # =================================================

            probabilities = (
                liveness_result.get(
                    "probabilities",
                    []
                )
            )

            raw_scores = (
                liveness_result.get(
                    "raw_scores",
                    []
                )
            )

            if len(probabilities) != 3:

                results.append({

                    "status":
                        "liveness_error",

                    "student_id":
                        student_id,

                    "student_name":
                        student_name,

                    "confidence":
                        recognition_confidence,

                    "error":
                        (
                            "Expected three liveness "
                            "probabilities."
                        ),

                    "face":
                        face
                })

                continue

            # -------------------------------------------------
            # INDIVIDUAL CLASS PROBABILITIES
            # -------------------------------------------------

            class_0 = float(
                probabilities[0]
            )

            class_1 = float(
                probabilities[1]
            )

            class_2 = float(
                probabilities[2]
            )

            # -------------------------------------------------
            # MODEL'S CURRENT WINNING CLASS
            # -------------------------------------------------

            predicted_class = int(
                liveness_result.get(
                    "predicted_class",
                    max(
                        range(3),
                        key=lambda index:
                            probabilities[index]
                    )
                )
            )

            predicted_probability = float(
                liveness_result.get(
                    "predicted_probability",
                    probabilities[
                        predicted_class
                    ]
                )
            )

            # =================================================
            # IMPORTANT
            # =================================================
            #
            # DO NOT ASSUME A REAL CLASS.
            #
            # For diagnostic purposes we expose the highest
            # probability as the currently predicted class.
            #
            # This is NOT yet a "live" decision.
            # =================================================

            # =================================================
            # 8. TEMPORAL HISTORY
            # =================================================

            student_key = str(
                student_id
            )

            if (
                student_key
                not in self.liveness_history
            ):

                self.liveness_history[
                    student_key
                ] = []

            history = self.liveness_history[
                student_key
            ]

            # -------------------------------------------------
            # STORE COMPLETE PROBABILITY VECTOR
            # -------------------------------------------------

            history.append({

                "class_0":
                    class_0,

                "class_1":
                    class_1,

                "class_2":
                    class_2,

                "predicted_class":
                    predicted_class,

                "predicted_probability":
                    predicted_probability
            })

            # -------------------------------------------------
            # LIMIT HISTORY SIZE
            # -------------------------------------------------

            if (
                len(history)
                > self.min_liveness_frames
            ):

                history.pop(0)

            # =================================================
            # 9. TEMPORAL AVERAGES
            # =================================================

            smoothed_class_0 = (
                sum(
                    item["class_0"]
                    for item in history
                )
                /
                len(history)
            )

            smoothed_class_1 = (
                sum(
                    item["class_1"]
                    for item in history
                )
                /
                len(history)
            )

            smoothed_class_2 = (
                sum(
                    item["class_2"]
                    for item in history
                )
                /
                len(history)
            )

            smoothed_probabilities = [

                smoothed_class_0,

                smoothed_class_1,

                smoothed_class_2
            ]

            # =================================================
            # 10. SMOOTHED WINNER
            # =================================================

            smoothed_predicted_class = int(
                max(
                    range(3),
                    key=lambda index:
                        smoothed_probabilities[
                            index
                        ]
                )
            )

            smoothed_predicted_probability = float(
                smoothed_probabilities[
                    smoothed_predicted_class
                ]
            )

            # =================================================
            # 11. TEMPORAL CONSISTENCY
            # =================================================

            predicted_classes = [

                item["predicted_class"]

                for item in history
            ]

            same_class_count = (
                predicted_classes.count(
                    smoothed_predicted_class
                )
            )

            # =================================================
            # 12. DIAGNOSTIC VERDICT
            # =================================================

            if (
                len(history)
                < self.min_liveness_frames
            ):

                verdict = (
                    "COLLECTING"
                )

            elif (
                same_class_count
                >= self.min_live_frames
            ):

                verdict = (
                    "STABLE_CLASS"
                )

            else:

                verdict = (
                    "UNSTABLE_CLASS"
                )

            # =================================================
            # 13. PRINT DIAGNOSTICS
            # =================================================

            print()

            print(
                "=" * 60
            )

            print(
                "[LIVENESS DIAGNOSTIC]"
            )

            print(
                f"Student: "
                f"{student_name}"
            )

            print(
                f"Recognition confidence: "
                f"{recognition_confidence:.2%}"
            )

            print()

            print(
                f"Raw scores: "
                f"{raw_scores}"
            )

            print()

            print(
                f"Class 0: "
                f"{class_0:.4f} "
                f"({class_0:.2%})"
            )

            print(
                f"Class 1: "
                f"{class_1:.4f} "
                f"({class_1:.2%})"
            )

            print(
                f"Class 2: "
                f"{class_2:.4f} "
                f"({class_2:.2%})"
            )

            print()

            print(
                f"Current predicted class: "
                f"{predicted_class}"
            )

            print(
                f"Current predicted probability: "
                f"{predicted_probability:.2%}"
            )

            print()

            print(
                "Smoothed probabilities:"
            )

            print(
                f"  Class 0: "
                f"{smoothed_class_0:.2%}"
            )

            print(
                f"  Class 1: "
                f"{smoothed_class_1:.2%}"
            )

            print(
                f"  Class 2: "
                f"{smoothed_class_2:.2%}"
            )

            print()

            print(
                f"Smoothed predicted class: "
                f"{smoothed_predicted_class}"
            )

            print(
                f"Smoothed confidence: "
                f"{smoothed_predicted_probability:.2%}"
            )

            print()

            print(
                f"Frames collected: "
                f"{len(history)}/"
                f"{self.min_liveness_frames}"
            )

            print(
                f"Stable winning-class frames: "
                f"{same_class_count}/"
                f"{len(history)}"
            )

            print(
                f"Diagnostic verdict: "
                f"{verdict}"
            )

            print()

            print(
                "IMPORTANT: "
                "No class is currently interpreted "
                "as LIVE."
            )

            print(
                "=" * 60
            )

            # =================================================
            # 14. RETURN RESULT
            # =================================================

            results.append({

                "status":
                    "liveness_diagnostic",

                "student_id":
                    student_id,

                "student_name":
                    student_name,

                "confidence":
                    recognition_confidence,

                "raw_scores":
                    raw_scores,

                "class_0":
                    class_0,

                "class_1":
                    class_1,

                "class_2":
                    class_2,

                "predicted_class":
                    predicted_class,

                "predicted_probability":
                    predicted_probability,

                "smoothed_class_0":
                    smoothed_class_0,

                "smoothed_class_1":
                    smoothed_class_1,

                "smoothed_class_2":
                    smoothed_class_2,

                "smoothed_predicted_class":
                    smoothed_predicted_class,

                "smoothed_predicted_probability":
                    smoothed_predicted_probability,

                "frames":
                    len(history),

                "required_frames":
                    self.min_liveness_frames,

                "stable_class_frames":
                    same_class_count,

                "min_stable_frames":
                    self.min_live_frames,

                "threshold":
                    self.real_threshold,

                "verdict":
                    verdict,

                "face":
                    face
            })

        return results

    # =========================================================
    # RESET SESSION
    # =========================================================

    def reset_session(self):
        """
        Reset all temporary session state.
        """

        self.liveness_history = {}

        self.recognized_students = set()

        self.logged_students = set()

        print(
            "Attendance engine session reset."
        )