from . import base
from llm.client import base as llm
from typing import List
from enum import Enum


class VerificationType(Enum):
    """A type representing the style in which to execute verifications"""

    JOINT = 0
    TWO_STEP = 1
    FACTORED = 2


DEFAULT_N_VERIFICATIONS = 5


class ChainOfVerification(base.Prompter):
    """
    A base class for different prompting techniques. Takes LLM
    client as an argument and provides a common interface to all the prompting classes.
    """

    def __init__(
        self,
        client: llm.AbstractLLM,
        execution_type: VerificationType = VerificationType.TWO_STEP,
    ) -> None:
        super().__init__(client)
        self._execution_type = execution_type

    def prompt(self, prompt: str) -> str:
        """
        This method prompts based upon the Chain of Verification technique
        outlined here: https://arxiv.org/pdf/2309.11495.pdf
        """
        # 1. Generate Baseline Response: Given a query, generate the response using the LLM.
        baseline_response = self._client.query(prompt)

        verifications = self._generate_verifications(prompt, baseline_response)

        if self._execution_type == VerificationType.JOINT:
            return self._generate_verified_response(verifications, baseline_response)

        inconsistancies = self._execute_verifications(baseline_response, verifications)

        return self._generate_verified_response(inconsistancies, baseline_response)

    def _generate_verifications(self, query: str, baseline_response: str) -> str:
        """
        2. Plan Verifications: Given both query and baseline response, generate a list of
        verification questions that could help to self-analyze if there are any mistakes
        in the original response.
        """

        verification_planning_prompt = f"""
            You are given the following query: {query}

            And the following response generated by a language model: {baseline_response}.

            For each fact provided in the response, generate a question that 
            can be used to verify the correctness of the fact.
        """

        if self._execution_type == VerificationType.JOINT:
            verification_planning_prompt += """
                From each of the questions generated, check the reponse to ensure that the information is accurate.
                Your final response should be a list of all the facts from the baseline response.
            """

        response = self._client.query(verification_planning_prompt)

        return response

    def _execute_verifications(self, response: str, verifications: str) -> str:
        """
        3. Execute Verifications: Answer each verification question in turn,
        and hence check the answer against the original response to check
        for inconsistencies or mistakes.
        """

        if self._execution_type == VerificationType.TWO_STEP:
            verification_execution_prompt = f"""
                Please provide answers to the following verification questions one at a time.
                
                {verifications}
                
                For the provided response
                
                {response}
            """

            hallucinations_detected = self._client.query(verification_execution_prompt)

            return hallucinations_detected

        elif self._execution_type == VerificationType.FACTORED:
            # TODO implement if two step is not performing well enough
            pass

        return ""

    def _generate_verified_response(self, inconsistancies: str, response: str) -> str:
        """
        4. Generate Final Verified Response: Given the discovered inconsistencies
        (if any), generate a revised response incorporating the verification results
        """

        final_verified_response_generator = f"""
            Given the following inconsistancies: {inconsistancies}
            
            And the following response: {response}
        
            Generate a final, revised response with all the fixes based on 
            the inconsistancies.
        """

        final_response = self._client.query(final_verified_response_generator)

        return final_response
