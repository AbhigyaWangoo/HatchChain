from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from memgpt import MemGPT

openai_client = OpenAI()

class AbstractClassifier(ABC):
    def __init__(self, hyperparams: List[str]) -> None:
        self._hyperparams = ', '.join(map(str, hyperparams))
        self._context_length = 4097
        self._memgpt_client = MemGPT(
            quickstart="openai",
        )

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    def _prompt_gpt(self, prompt: str, engine: str = "gpt-3.5-turbo-16k", temperature: int = 0.2) -> str:
        """ A simple wrapper to the gpt api """

        response = openai_client.chat.completions.create(
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
    
    def _prompt_hatch_persona(self, prompt: str):
        # You can set many more parameters, this is just a basic example
        agent_id = self._memgpt_client.create_agent(
            agent_config={
            "persona": "hatch_persona",
            "human": "hatch_human",
            }
        )

        # Now that we have an agent_name identifier, we can send it a message!
        # The response will have data from the MemGPT agent
        response = self._memgpt_client.user_message(agent_id=agent_id, message=prompt)
        return response[2]["assistant_message"]
