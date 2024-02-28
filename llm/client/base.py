from abc import ABC, abstractmethod


class AbstractLLM(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass


def get_navigation_string(heuristic: str, input_str: str, category: str) -> str:
    """Returns a crafted heuristic prompt based on the provided args"""

    return f"""
        You have a candidate and a label. On the bases of the following heuristcs
        here: {heuristic} decide whether the following candidate: {input_str} fits the category 
        of {category}. When providing a reasoning, only reference the specific heuristics provided,
        all your lines of reasoning should be relevant to the provided heuristic.
        
        Your output should always be modelled as follows:
        reject | accept:<reasoning for why the candidate should be accepted or rejected>
        """
