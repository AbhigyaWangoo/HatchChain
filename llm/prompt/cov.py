from . import base
from llm.client import base as llm
from typing import List


class ChainOfVerification(base.Prompter):
    """
    A base class for different prompting techniques. Takes LLM
    client as an argument and provides a common interface to all the prompting classes.
    """

    def __init__(self, client: llm.AbstractLLM) -> None:
        super().__init__(client)

    def prompt(self, prompt: str) -> str:
        """
        This method prompts based upon the Chain of Verification technique
        outlined here: https://arxiv.org/pdf/2309.11495.pdf
        """
        # 1. Generate Baseline Response: Given a query, generate the response using the LLM.
        pass

    def _generate_verifications(self, query: str, baseline_response: str) -> List[str]:
        """
        2. Plan Verifications: Given both query and baseline response, generate a list of verification
        questions that could help to self-analyze if there are any mistakes in the original response.
        """
        pass

    def _execute_verifications(self, response: str, verifications: List[str]) -> str:
        """
        3. Execute Verifications: Answer each verification question in turn, and hence check the answer
        against the original response to check for inconsistencies or mistakes.
        """
        pass

    def _generate_verified_response(self, inconsistancies: str, response: str) -> str:
        """
        4. Generate Final Verified Response: Given the discovered inconsistencies (if any), generate a
        revised response incorporating the verification results
        """
        pass
