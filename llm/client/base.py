from abc import ABC, abstractmethod


class AbstractLLM(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass
