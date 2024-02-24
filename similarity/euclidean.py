from numpy import ndarray, linalg
from . import base


class EuclideanSimilarity(base.Similarity):
    """
    A class for calculating euclidean similarity
    """

    def compute_similarity(self, doc1: ndarray, doc2: ndarray) -> float:
        # Compute Euclidean similarity between doc1 and doc2
        euclidean_distance = linalg.norm(doc1 - doc2)

        # Convert distance to similarity score (1 / (1 + distance))
        similarity_score = 1 / (1 + euclidean_distance)

        return similarity_score
