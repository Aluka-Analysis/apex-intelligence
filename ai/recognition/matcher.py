"""
Face Matching Module.

Compares a detected face against enrolled student embeddings
and returns the best identity match.

The enrollment system stores multiple embeddings per student.
The matcher therefore supports:

    Student
        ├── Embedding 1
        ├── Embedding 2
        ├── Embedding 3
        ├── Embedding 4
        └── Embedding 5

The query face is compared against every stored embedding,
and the strongest similarity score is used for identification.
"""

import numpy as np

from ai.recognition.embedder import FaceEmbedder


class FaceMatcher:
    """
    Matches a detected face against enrolled students.

    Supports multiple face embeddings per student.
    """

    def __init__(self, threshold: float = 0.45):
        """
        Initialize the matcher.

        Args:
            threshold:
                Minimum cosine similarity required
                to consider a face a match.

        IMPORTANT:
            This is an initial experimental threshold.
            It should NOT be considered production-calibrated.
        """

        self.threshold = threshold
        self.embedder = FaceEmbedder()

    def match(
        self,
        query_embedding: np.ndarray,
        enrolled_students: list
    ) -> dict:
        """
        Match a query face against enrolled students.

        Expected input structure:

        [
            {
                "student_id": "...",
                "student_name": "Chidi Okafor",
                "class_name": "JSS 2A",
                "embeddings": [
                    "...",
                    "...",
                    "..."
                ]
            }
        ]

        Returns:

        {
            "matched": True/False,
            "student_id": "...",
            "student_name": "...",
            "confidence": 0.82,
            "status": "matched"
        }
        """

        if not enrolled_students:
            return {
                "matched": False,
                "student_id": None,
                "student_name": None,
                "confidence": 0.0,
                "status": "no_embeddings"
            }

        query_norm = self.embedder.normalize(
            query_embedding
        )

        best_score = -1.0
        best_student = None

        # -------------------------------------------------
        # Compare query against every enrolled student
        # -------------------------------------------------

        for student in enrolled_students:

            embeddings = student.get("embeddings", [])

            if not embeddings:
                continue

            # ---------------------------------------------
            # Compare against every enrollment sample
            # ---------------------------------------------

            for embedding_json in embeddings:

                try:

                    stored_embedding = (
                        self.embedder.from_json(
                            embedding_json
                        )
                    )

                    stored_norm = (
                        self.embedder.normalize(
                            stored_embedding
                        )
                    )

                    score = (
                        self.embedder.calculate_similarity(
                            query_norm,
                            stored_norm
                        )
                    )

                    # Keep strongest embedding
                    if score > best_score:

                        best_score = score
                        best_student = student

                except Exception as error:

                    print(
                        f"Warning: could not process "
                        f"embedding for "
                        f"{student.get('student_name')}: "
                        f"{error}"
                    )

        # -------------------------------------------------
        # No valid embeddings
        # -------------------------------------------------

        if best_student is None:

            return {
                "matched": False,
                "student_id": None,
                "student_name": None,
                "confidence": 0.0,
                "status": "no_valid_embeddings"
            }

        confidence = float(
            max(0.0, best_score)
        )

        # -------------------------------------------------
        # Match accepted
        # -------------------------------------------------

        if confidence >= self.threshold:

            return {
                "matched": True,
                "student_id": best_student["student_id"],
                "student_name": best_student["student_name"],
                "class_name": best_student.get("class_name"),
                "confidence": round(
                    confidence,
                    4
                ),
                "status": "matched"
            }

        # -------------------------------------------------
        # Unknown face
        # -------------------------------------------------

        return {
            "matched": False,
            "student_id": None,
            "student_name": None,
            "class_name": None,
            "confidence": round(
                confidence,
                4
            ),
            "status": "unknown"
        }

    def match_best(
        self,
        query_embedding: np.ndarray,
        enrolled_students: list,
        top_k: int = 3
    ) -> list:
        """
        Return the strongest candidate matches.

        Useful for debugging, threshold analysis,
        and laboratory experiments.

        Returns:

        [
            {
                "student_id": "...",
                "student_name": "...",
                "confidence": 0.82,
                "matched": True
            }
        ]
        """

        if not enrolled_students:
            return []

        query_norm = self.embedder.normalize(
            query_embedding
        )

        candidates = []

        for student in enrolled_students:

            embeddings = student.get(
                "embeddings",
                []
            )

            student_best_score = -1.0

            for embedding_json in embeddings:

                try:

                    stored_embedding = (
                        self.embedder.from_json(
                            embedding_json
                        )
                    )

                    stored_norm = (
                        self.embedder.normalize(
                            stored_embedding
                        )
                    )

                    score = (
                        self.embedder.calculate_similarity(
                            query_norm,
                            stored_norm
                        )
                    )

                    if score > student_best_score:
                        student_best_score = score

                except Exception:
                    continue

            if student_best_score >= 0:

                candidates.append({
                    "student_id": student["student_id"],
                    "student_name": student["student_name"],
                    "class_name": student.get("class_name"),
                    "confidence": round(
                        float(student_best_score),
                        4
                    ),
                    "matched": (
                        student_best_score
                        >= self.threshold
                    )
                })

        candidates.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return candidates[:top_k]

    def set_threshold(
        self,
        threshold: float
    ):
        """
        Dynamically change the recognition threshold.

        Useful during experimentation.

        Example:

            matcher.set_threshold(0.50)
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0.0 and 1.0"
            )

        self.threshold = threshold

        print(
            f"Recognition threshold updated to "
            f"{threshold:.2%}"
        )

    def get_threshold(self) -> float:
        """Return current recognition threshold."""

        return self.threshold