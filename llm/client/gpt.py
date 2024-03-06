from openai import OpenAI
from . import base


class GPTClient(base.AbstractLLM):
    """A client module to call the GPT API"""

    def __init__(self) -> None:
        super().__init__()

        self._client = OpenAI()

    def query(self, prompt: str, engine: str = "gpt-4", temperature: int = 0.2) -> str:
        """A simple wrapper to the gpt api"""

        response = self._client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=engine,
            temperature=temperature,
        )

        generated_response = response.choices[0].message.content.strip()

        return generated_response
