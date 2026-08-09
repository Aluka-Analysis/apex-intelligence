"""
Face Matching Module.
Compares a detected face against all enrolled students
and returns the best match above a confidence threshold.

This is the core identity verification component.
"""

import numpy as np
from ai.recognition.embedder import FaceEmbedder


class FaceMatcher:
    """
    Matches a detected face against a database of
    enrolled student embeddings.
    """

    def __init__(self, threshold: float = 0.45):
        """
        Initialize the matcher.

        Args:
            threshold: minimum similarity score to accept as a match.
                      Below this score the face is considered unknown.
                      0.45 is a balanced starting point.
                      Higher = stricter = fewer false accepts
                      Lower  = looser  = fewer false rejects
        """
        self.threshold = threshold
        self.embedder  = FaceEmbedder()

    def match(
        self,
        query_embedding: np.ndarray,
        enrolled_embeddings: list
    ) -> dict:
        """
        Match a query embedding against all enrolled embeddings.

        Args:
            query_embedding:    embedding of face to identify
            enrolled_embeddings: list of dicts with keys:
                                 student_id, student_name, embedding

        Returns:
            Dict with keys:
            - matched:      True if match found
            - student_id:   ID of matched student or None
            - student_name: Name of matched student or None
            - confidence:   similarity score
            - status:       'matched', 'unknown', or 'no_embeddings'
        """
        if not enrolled_embeddings:
            return {
                'matched':      False,
                'student_id':   None,
                'student_name': None,
                'confidence':   0.0,
                'status':       'no_embeddings'
            }

        query_norm = self.embedder.normalize(query_embedding)

        best_score    = 0.0
        best_match    = None

        for enrolled in enrolled_embeddings:
            stored_embedding = self.embedder.from_json(
                enrolled['embedding']
            )
            stored_norm = self.embedder.normalize(stored_embedding)

            score = self.embedder.calculate_similarity(
                query_norm,
                stored_norm
            )

            if score > best_score:
                best_score = score
                best_match = enrolled

        if best_score >= self.threshold and best_match:
            return {
                'matched':      True,
                'student_id':   best_match['student_id'],
                'student_name': best_match['student_name'],
                'confidence':   round(best_score, 4),
                'status':       'matched'
            }

        return {
            'matched':      False,
            'student_id':   None,
            'student_name': None,
            'confidence':   round(best_score, 4),
            'status':       'unknown'
        }

    def match_best(
        self,
        query_embedding: np.ndarray,
        enrolled_embeddings: list,
        top_k: int = 3
    ) -> list:
        """
        Return top K matches for debugging and analysis.
        Useful in the laboratory notebook to understand
        model behaviour.

        Args:
            query_embedding:    embedding to match
            enrolled_embeddings: enrolled students list
            top_k:              number of top matches to return

        Returns:
            List of top K matches sorted by confidence
        """
        if not enrolled_embeddings:
            return []

        query_norm = self.embedder.normalize(query_embedding)
        scores = []

        for enrolled in enrolled_embeddings:
            stored_embedding = self.embedder.from_json(
                enrolled['embedding']
            )
            stored_norm = self.embedder.normalize(stored_embedding)
            score = self.embedder.calculate_similarity(
                query_norm, stored_norm
            )
            scores.append({
                'student_id':   enrolled['student_id'],
                'student_name': enrolled['student_name'],
                'confidence':   round(score, 4),
                'matched':      score >= self.threshold
            })

        scores.sort(key=lambda x: x['confidence'], reverse=True)
        return scores[:top_k]