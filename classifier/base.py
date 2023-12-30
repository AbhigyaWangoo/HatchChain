from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI

client = OpenAI()


class AbstractClassifier(ABC):
    def __init__(self, hyperparams: List[str]) -> None:
        self._hyperparams = ', '.join(map(str, hyperparams))
        self._context_length = 4097

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    def _prompt_gpt(self, prompt: str, engine: str = "gpt-3.5-turbo-16k", temperature: int = 0.2) -> str:
        """ A simple wrapper to the gpt api """

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=engine,
            temperature=temperature
        )

        generated_response = response.choices[0].message.content.strip()

        return generated_response
