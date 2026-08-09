"""
Face Embedding Module.
Converts detected faces into numerical vectors
that represent identity.

A face embedding is a 512-dimension vector.
Two photos of the same person will have
similar vectors regardless of lighting or angle.
Two photos of different people will have
very different vectors.
"""

import numpy as np
import json


class FaceEmbedder:
    """
    Extracts and manages face embeddings.
    Works with InsightFace detected face objects.
    """

    def extract(self, face) -> np.ndarray:
        """
        Extract embedding vector from a detected face.

        Args:
            face: InsightFace detected face object

        Returns:
            512-dimension numpy array
        """
        if face is None:
            raise ValueError("No face provided for embedding extraction")

        if not hasattr(face, 'embedding'):
            raise ValueError("Face object has no embedding attribute")

        return face.embedding

    def to_json(self, embedding: np.ndarray) -> str:
        """
        Convert embedding to JSON string for database storage.

        Args:
            embedding: 512-dimension numpy array

        Returns:
            JSON string representation
        """
        return json.dumps(embedding.tolist())

    def from_json(self, embedding_json: str) -> np.ndarray:
        """
        Convert JSON string back to numpy array.

        Args:
            embedding_json: JSON string from database

        Returns:
            512-dimension numpy array
        """
        return np.array(json.loads(embedding_json))

    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Returns a score between 0 and 1.
        Higher score means more similar faces.

        Args:
            embedding1: first face embedding
            embedding2: second face embedding

        Returns:
            Similarity score between 0.0 and 1.0
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp to 0-1 range
        return float(np.clip(similarity, 0.0, 1.0))

    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length.
        Improves similarity calculation consistency.

        Args:
            embedding: raw face embedding

        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm