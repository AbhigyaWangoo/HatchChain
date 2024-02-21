from . import base
from numpy import ndarray, dot, linalg


class CosineSimilarity(base.Similarity):
    def __init__(self) -> None:
        super().__init__()

    def compute_similarity(self, doc1: ndarray, doc2: ndarray) -> float:
        # Ensure both input arrays are not empty
        if len(doc1) == 0 or len(doc2) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Compute the dot product of the two arrays
        dot_product = dot(doc1, doc2)

        # Compute the Euclidean norms of the two arrays
        norm_doc1 = linalg.norm(doc1)
        norm_doc2 = linalg.norm(doc2)

        if norm_doc2 == 0 or norm_doc1 == 0:
            return 0.0

        # Calculate the cosine similarity
        similarity = dot_product / (norm_doc1 * norm_doc2)

        return similarity
