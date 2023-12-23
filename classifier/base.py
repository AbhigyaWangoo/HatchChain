from abc import ABC, abstractmethod
from typing import List
import openai

class AbstractClassifier(ABC):
    def __init__(self, hyperparams: List[str]) -> None:
        self._hyperparams = ', '.join(map(str, hyperparams))
        self._context_length=4097

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    def _prompt_gpt(self, prompt: str, tokenlimit: int = 1000, engine: str = "text-davinci-003", temperature: int = 0.7) -> str:
        """ A simple wrapper to the gpt api """

        completion = openai.Completion.create(
            prompt=prompt,
            engine=engine,
            max_tokens=tokenlimit,
            temperature=temperature
        )

        generated_response = completion['choices'][0]['text'].strip()

        return generated_response
