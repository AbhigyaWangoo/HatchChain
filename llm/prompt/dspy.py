from . import base as prompter
from llm.client import base as llm
import dspy
from dspy import Example
from typing import Tuple, List
import json

JOB_CONTEXT="heuristic"
OUTPUT="navigation"
INPUT_RESUME="input resume"

class DSPyPrompter(prompter.Prompter):
    """
    A class for DSPy prompting. Based off paper
    """

    def __init__(self, client: llm.AbstractLLM) -> None:
        self._client = client

    def prompt(self, prompt: str) -> str:
        """
        The abstract method for a prompter to execute a prompt
        """
        pass

    def load_dataset(self, dataset: str) -> Tuple[List[Example], List[Example]]:
        """
        Returns a (train, test) dataset.
        """

        with open(dataset, "r", encoding="utf8") as fp:
            corpus = json.load(fp)
            for item in corpus:
                context = item[JOB_CONTEXT]
                example_output = item[OUTPUT]
                input_resume = item[INPUT_RESUME]

                # Example(question=)


class DSpyCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
        self.turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=250)
        dspy.settings.configure(lm=self.turbo)

    def forward(self, question):
        return self.prog(question=question)
