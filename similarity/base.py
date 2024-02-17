from abc import abstractmethod, ABC
from numpy import ndarray, argsort
from typing import Dict

from ordereddict import OrderedDict

# Ideally this should be evaluating HatchVectors not ndarrays. That requires us to make a Vector parent class, with
# JobVector and ResumeVector children classes. Not doin allat rn tho.


class Similarity(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute_similarity(self, doc1: ndarray, doc2: ndarray) -> float:
        pass

    def top_k(self, v1: ndarray, v_dict: Dict[int, ndarray], k: int) -> OrderedDict[int, ndarray]:
        # Ensure the input vector is not empty
        if len(v1) == 0:
            raise ValueError("Input vector cannot be empty")

        # Calculate similarities between v1 and all vectors in v_dict
        similarities = {key: self.compute_similarity(v1, vector)
                        for key, vector in v_dict.items()}

        # Sort the dictionary by similarity in descending order
        sorted_similarities = dict(
            sorted(similarities.items(), key=lambda item: item[1], reverse=True))

        # Get the top k similar vectors
        top_k_vectors = {key: v_dict[key]
                         for key in list(sorted_similarities.keys())[:k]}

        return OrderedDict(top_k_vectors)
