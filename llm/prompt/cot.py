from . import base as prompter
from llm.client import base as llm

class ChainOfThoughtPrompter(prompter.Prompter):
    """
    A class for chain of thought prompting. Based off paper 
    """

    def __init__(self, client: llm.AbstractLLM) -> None:
        self._client = client

    def prompt(self, prompt: str) -> str:
        """ 
        The abstract method for a prompter to execute a prompt 
        """
        pass
