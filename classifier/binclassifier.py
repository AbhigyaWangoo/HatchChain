from typing import Tuple, Dict, List
from openai import OpenAI

client = OpenAI()
from . import base


class ExplainableClassifier(base.AbstractClassifier):
    """
    This is a classifier that uses LLMs to generate explanations for
    binary classifications.
    """

    def __init__(self, hyperparams: List[str], provided_context: str) -> None:
        self._user_context = provided_context
        super().__init__(hyperparams=hyperparams)

    def classify(self, input: str, category: str) -> Tuple[bool, str]:
        approve_reasons = self._prompt_gpt(self._get_context(category, input))
        reject_reasons = self._prompt_gpt(self._get_context(category, input, False))

        print(approve_reasons)
        print(reject_reasons)

        return self._classify_from_reasons(approve_reasons, reject_reasons, category)

    def _get_context(self, category: str, input_data: str, approve: bool = True) -> str:
        approve_or_deny = "reject"
        if approve:
            approve_or_deny = "accept"

        context = f"""You are an explainable binary classifier. Using the following parameters the recruiter cares
        about regarding a candidate: {self._hyperparams}, for each parameter, generate a precise reason to {approve_or_deny} 
        the candidate for the category of {category}. If there is no reason to {approve_or_deny} the candidate for that hyperparameter,
        simply do not output a line for that paramenter. Your reasonings must, at all costs, be relevant to the category provided.

        candidate data: {input_data}

        output format:
        <parameter 1>:reasoning for {approve_or_deny}
        <parameter 2>:reasoning for {approve_or_deny}
        """

        return context

    def _simple_classify(self, input: str, category: str):
        output_prompt = f"""
        Given the following resume, should the candidate be accepted or rejected for the position of {category}? Provide 
        an explanation for why they should or shouldn't be. The output format is below as well.
        output format:
        True | False:<reasoning paragraph>
        resume: {input}
        """

        completion = client.completions.create(
            prompt=output_prompt, engine="text-davinci-003"
        )

        generated_response = completion["choices"][0]["text"].strip()
        decision, *reason = generated_response.split(":")

        if decision.lower() == "false":
            return False, " ".join(reason)
        return True, " ".join(reason)

    def _summarize_chunk(self, chunk: str, cutoff: int = 2000) -> str:
        """Helper function to truncate a string at a specific point and recursively summarize it"""
        summarizer_prompt = f"Summarize the following into exactly {str(cutoff)} characters or less: {chunk}"

        completion = client.completions.create(
            prompt=summarizer_prompt, engine="text-davinci-003", max_tokens=cutoff
        )

        generated_response = completion["choices"][0]["text"].strip()

        return generated_response

    def _trim_input(
        self, input: str, n_chunks: int = 5, reduction_multiplier: float = 0.75
    ) -> str:
        whitespace_removal = input.replace(r" {2,}", "").strip()

        def split_string_into_chunks(s: str, n: int) -> List[str]:
            """Split a string into n equal chunks."""
            if n <= 0:
                raise ValueError("Number of chunks (n) must be greater than 0.")
            chunk_size = len(s) // n
            remainder = len(s) % n
            chunks = [s[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]
            # Add the remaining characters to the last chunk
            if remainder:
                chunks[-1] += s[-remainder:]
            return chunks

        chunks = split_string_into_chunks(whitespace_removal, n_chunks)
        final = ""
        for chunk in chunks:
            final += self._summarize_chunk(
                chunk, int((len(whitespace_removal) // n_chunks) * reduction_multiplier)
            )

        return final

    def _classify_from_reasons(
        self, approve_reasons: List[str], reject_reasons: List[str], role: str
    ) -> Tuple[bool, str]:
        """Provided with the reasons to approve or deny a classification, and a reasoning for said classification"""
        final_classification_context = f"""
        I have provided you with the following reasons to approve or deny a candidate for the role of {role}.
        Decide whether to approve or deny this candidate, and specify why exactly that candidate should be approved or denied.
        If you cannot decide, make an educated guess, do not say you cannot decide.
        
        Reasons to approve:
        {approve_reasons}
        Reasons to reject:
        {reject_reasons}

        your output should be in the following format:
        <true | false>:<paragraph formatted reasoning>
        """

        completion = client.completions.create(
            prompt=final_classification_context, engine="text-davinci-003"
        )

        generated_response = completion["choices"][0]["text"].strip()
        decision, *reasons = generated_response.split(":")
        print("Decision on candidate:" + decision)
        return bool(decision), " ".join(reasons)
