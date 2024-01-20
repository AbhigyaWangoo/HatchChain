from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from memgpt import MemGPT
from ..llm import gpt, memgpt

class AbstractClassifier(ABC):
    def __init__(self, hyperparams: List[str]) -> None:
        self._hyperparams = ', '.join(map(str, hyperparams))
        self._context_length = 4097
        self._memgpt_client = memgpt.MemGPTClient()
        self._openai_client = gpt.GPTClient()

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    def _prompt_gpt(self, prompt: str, engine: str = "gpt-3.5-turbo-16k", temperature: int = 0.2) -> str:
        """ A simple wrapper to the gpt client """
        return self._openai_client.query(prompt, engine, temperature)
    
    def _prompt_hatch_persona(self, prompt: str):
        """ A simple wrapper to the memgpt client """
        # You can set many more parameters, this is just a basic example
        return self._memgpt_client.query(prompt)