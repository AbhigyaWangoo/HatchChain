from . import base as prompter
from llm.client import base as llm

class FewShotPrompter(prompter.Prompter):
    """
    A class for few shot prompting.
    """

    def __init__(self, client: llm.AbstractLLM) -> None:
        self._client = client

    def prompt(self, prompt: str) -> str:
        """ 
        The abstract method for a prompter to execute a prompt 
        """
        pass
