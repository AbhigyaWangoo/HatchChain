from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from llm.client import gpt, runpod
from llm.prompt import few_shot, cot


class AbstractClassifier(ABC):
    def __init__(self, hyperparams: List[str]) -> None:
        self._hyperparams = ", ".join(map(str, hyperparams))
        self._context_length = 4097
        self._openai_client = gpt.GPTClient()
        self._runpod_client = runpod.RunPodClient()

        self._prompter = cot.ChainOfThoughtPrompter(self._runpod_client)

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    @abstractmethod
    def save_model(self, path: str) -> Dict[Any, Any]:
        """Saves the model to file, and returns the json created and saved to the file."""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Reads the model from the provided file into the current instance"""
        pass

    def _prompt_gpt(
        self, prompt: str, engine: str = "gpt-3.5-turbo-16k", temperature: int = 0.2
    ) -> str:
        """A simple wrapper to the gpt client"""
        return self._openai_client.query(prompt, engine, temperature)

    def _prompt_runpod(self, prompt: str, full_resp: bool = False):
        """A simple wrapper to the runpod client"""
        return self._runpod_client.query(prompt, full_resp)
