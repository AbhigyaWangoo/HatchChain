from typing import Tuple, Dict, List
import openai

class ExplainableClassifier():
    """
    This is a classifier that uses LLMs to generate explanations for 
    binary classifications.
    """

    def __init__(self, hyperparams: List[str], provided_context: str) -> None:
        self._user_context = provided_context
        self._hyperparams = hyperparams

    def __init__(self, hyperparams: List[str]) -> None:
        self._hyperparams = hyperparams
        hyperparam_str = ', '.join(map(str, hyperparams))
        self._resume_context = f"""
        You are an explainable binary classifier. Using the following parameters the recruiter cares
        about regarding a candidate: {hyperparam_str}, for each parameter, generate a precise reason to accept the candidate,
        and a precise reason to reject the candidate for the following category.
        """

        self._output_format = """
        Your output should be in the following format:
        <parameter 1>:<accept reasoning for parameter 1>;<reject reasoning for parameter 1>
        <parameter 2>:<accept reasoning for parameter 2>;<reject reasoning for parameter 2>
        """

    def classify(self, input: str, category: str) -> Tuple[bool, str]:
        initial_prompt = self._resume_context + \
            category + f"\n{input}" + self._output_format
        approve_deny_reasons = self._llm_reasoning_generation(initial_prompt)

        return self._classify_from_reasons(approve_deny_reasons, category)

    def _llm_reasoning_generation(self, input: str) -> str:
        """ Generates, given an input, a set of reasons from the context and input data. Should be outputted in provided format. """

        completion = openai.Completion.create(
            prompt=input,
            engine="text-davinci-003",
            max_tokens=1000,
        )
        
        generated_response = completion['choices'][0]['text'].strip()

        return generated_response

    def _classify_from_reasons(self, reasons: List[str], role: str) -> Tuple[bool, str]:
        """ Provided with the reasons to approve or deny a classification, and a reasoning for said classification """
        final_classification_context = f"""
        I have provided you with the following reasons to approve or deny a candidate for the role of {role}.
        Decide whether to approve or deny this candidate, and specify why exactly that candidate should be approved or denied.
        If you cannot decide, make an educated guess, do not say you cannot decide.

        {reasons}

        your output should be in the following format:
        <true | false>:<reasoning>
        """

        completion = openai.Completion.create(
            prompt=final_classification_context,
            engine="text-davinci-003",
            max_tokens=1000,
        )
        
        generated_response = completion['choices'][0]['text'].strip()
        decision, *reasons = generated_response.split(":")

        return bool(decision), " ".join(reasons)
