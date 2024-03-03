from . import base
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["MISTRAL_API_KEY"]
MODEL = "open-mixtral-8x7b"
LARGE_MODEL = "mistral-large-latest"


class MistralLLMClient(base.AbstractLLM):
    """A client module to call the mistral API"""

    def __init__(self) -> None:
        super().__init__()
        self._client = MistralClient(api_key=API_KEY)

    def query(
        self, prompt: str, model: str = MODEL, temperature: int = 0.2
    ) -> str:
        """A simple wrapper to the mistral api"""

        messages = [ChatMessage(role="user", content=prompt)]

        # No streaming
        chat_response = self._client.chat(
            model=model, messages=messages, temperature=temperature
        )

        return chat_response.choices[0].message.content
