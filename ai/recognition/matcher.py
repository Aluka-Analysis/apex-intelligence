"""
Face Matching Module.
Compares a detected face against all enrolled students
and returns the best match above a confidence threshold.
"""

import numpy as np
from ai.recognition.embedder import FaceEmbedder


class FaceMatcher:
    """
    Matches a detected face against enrolled student embeddings.
    """

    def __init__(self, threshold: float = 0.45):
        """
        Initialize the matcher.

        Args:
            threshold: minimum similarity score to accept as a match.
                      0.45 is a balanced starting point.
                      Higher = stricter = fewer false accepts
                      Lower  = looser  = fewer false rejects
        """
        self.threshold = threshold
        self.embedder  = FaceEmbedder()

    def match(
        self,
        query_embedding: np.ndarray,
        enrolled_students: list
    ) -> dict:
        """
        Match a query embedding against all enrolled students.
        Each student may have multiple embeddings.
        Returns the best match across all embeddings.

        Args:
            query_embedding:   embedding of face to identify
            enrolled_students: list of dicts with keys:
                               student_id, student_name,
                               class_name, embeddings (list)

        Returns:
            Dict with match result:
            - matched:      True if match found
            - student_id:   ID of matched student or None
            - student_name: Name of matched student or None
            - class_name:   Class of matched student or None
            - confidence:   similarity score
            - status:       matched, unknown, or no_enrollments
        """
        if not enrolled_students:
            return {
                'matched':      False,
                'student_id':   None,
                'student_name': None,
                'class_name':   None,
                'confidence':   0.0,
                'status':       'no_enrollments'
            }

        query_norm   = self.embedder.normalize(query_embedding)
        best_score   = 0.0
        best_student = None

        for student in enrolled_students:
            # Compare against ALL embeddings for this student
            # Take the highest score across all captures
            for embedding_json in student['embeddings']:
                stored      = self.embedder.from_json(embedding_json)
                stored_norm = self.embedder.normalize(stored)
                score       = self.embedder.calculate_similarity(
                    query_norm,
                    stored_norm
                )

                if score > best_score:
                    best_score   = score
                    best_student = student

        if best_score >= self.threshold and best_student:
            return {
                'matched':      True,
                'student_id':   best_student['student_id'],
                'student_name': best_student['student_name'],
                'class_name':   best_student['class_name'],
                'confidence':   round(best_score, 4),
                'status':       'matched'
            }

        return {
            'matched':      False,
            'student_id':   None,
            'student_name': None,
            'class_name':   None,
            'confidence':   round(best_score, 4),
            'status':       'unknown'
        }

    def match_top_k(
        self,
        query_embedding: np.ndarray,
        enrolled_students: list,
        top_k: int = 3
    ) -> list:
        """
        Return top K matches for debugging and analysis.
        Useful in the laboratory notebook to understand
        model behaviour.

        Args:
            query_embedding:   embedding to match
            enrolled_students: enrolled students list
            top_k:             number of top matches to return

        Returns:
            List of top K matches sorted by confidence
        """
        if not enrolled_students:
            return []

        query_norm = self.embedder.normalize(query_embedding)
        scores     = []

        for student in enrolled_students:
            best_score_for_student = 0.0

            for embedding_json in student['embeddings']:
                stored      = self.embedder.from_json(embedding_json)
                stored_norm = self.embedder.normalize(stored)
                score       = self.embedder.calculate_similarity(
                    query_norm,
                    stored_norm
                )
                if score > best_score_for_student:
                    best_score_for_student = score

            scores.append({
                'student_id':   student['student_id'],
                'student_name': student['student_name'],
                'class_name':   student['class_name'],
                'confidence':   round(best_score_for_student, 4),
                'matched':      best_score_for_student >= self.threshold
            })

        scores.sort(key=lambda x: x['confidence'], reverse=True)
        return scores[:top_k]