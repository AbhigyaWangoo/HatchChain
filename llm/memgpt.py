from memgpt import MemGPT
from . import base

class MemGPTClient(base.AbstractLLM):
    """ A client module to call the GPT API """
    def __init__(self, quickstart: str="openai") -> None:
        super().__init__()

        self._client = MemGPT(
            quickstart=quickstart,
        )

    def query(self, prompt: str, engine: str = "gpt-3.5-turbo-16k", temperature: int = 0.2) -> str:
        """ A simple wrapper to the gpt api """
        # You can set many more parameters, this is just a basic example
        agent_id = self._client.create_agent(
            agent_config={
            "persona": "hatch_persona",
            "human": "hatch_human",
            }
        )

        # Now that we have an agent_name identifier, we can send it a message!
        # The response will have data from the MemGPT agent
        response = self._client.user_message(agent_id=agent_id, message=prompt)
        return response[2]["assistant_message"]
