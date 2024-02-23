from . import base as prompter
from llm.client import base as llm
from enum import Enum

ZERO_SHOT_PROMPT="Let's think step by step"

class CotType(Enum):
    """ Enum for type of COT """
    ZERO_SHOT=0
    MANUAL_COT=1

class ChainOfThoughtPrompter(prompter.Prompter):
    """
    A class for chain of thought prompting. Based off paper 
    """

    def __init__(self, client: llm.AbstractLLM, type: CotType=CotType.ZERO_SHOT) -> None:
        self._client = client
        self._type = type

    def __set_cot_examples(self, prompt: str):
        """ TODO A helper function to set the cot examples """
        return prompt

    def prompt(self, prompt: str) -> str:
        """ 
        The abstract method for a prompter to execute a prompt 
        """
        formatted_prompt = "Q: " + prompt + "\n" + "A:"

        if self._type == CotType.ZERO_SHOT:
            formatted_prompt += ZERO_SHOT_PROMPT
        elif self._type == CotType.MANUAL_COT:
            formatted_prompt += self.__set_cot_examples(prompt)

        return self._client.query(formatted_prompt)
