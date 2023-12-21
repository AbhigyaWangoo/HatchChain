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
        self._context_length=4097

    def __init__(self, hyperparams: List[str]) -> None:
        self._hyperparams = hyperparams
        hyperparam_str = ', '.join(map(str, hyperparams))
        self._resume_context = f"""You are an explainable binary classifier. Using the following parameters the recruiter cares
        about regarding a candidate: {hyperparam_str}, for each parameter, generate a precise reason to accept the candidate,
        and a precise reason to reject the candidate for the provided category. Your reasonings must, at all costs, be relevant 
        to the category provided.
        """

        self._output_format = """Your output should be in the following format:
        <parameter 1>:<accept reasoning for parameter 1>;<reject reasoning for parameter 1>
        <parameter 2>:<accept reasoning for parameter 2>;<reject reasoning for parameter 2>
        """
        self._context_length=4097

    def classify(self, input: str, category: str) -> Tuple[bool, str]:
        cleaned_input = input
        # if len(input) > self._context_length:
        #     cleaned_input = self._trim_input(cleaned_input)
        
        # initial_prompt = self._resume_context + \
        #     category + f"\n{cleaned_input}" + self._output_format
        # approve_deny_reasons = self._llm_reasoning_generation(initial_prompt)
        
        # print(approve_deny_reasons)
        # print(category)
        
        # return self._classify_from_reasons(approve_deny_reasons, category)
        return self._simple_classify(cleaned_input, category)

    def _simple_classify(self, input: str, category: str):
        output_prompt = f"""
        Given the following resume, should the candidate be accepted or rejected for the position of {category}? Provide 
        an explanation for why they should or shouldn't be. The output format is below as well.
        output format:
        True | False:<reasoning paragraph>
        resume: {input}
        """

        completion = openai.Completion.create(
            prompt=output_prompt,
            engine="text-davinci-003"
        )

        generated_response = completion['choices'][0]['text'].strip()
        decision, *reason = generated_response.split(":")

        if decision.lower() == "false":
            return False, " ".join(reason)
        return True, " ".join(reason)

    def _summarize_chunk(self, chunk: str, cutoff: int = 2000) -> str:
        """Helper function to truncate a string at a specific point and recursively summarize it"""
        summarizer_prompt = f"Summarize the following into exactly {str(cutoff)} characters or less: {chunk}"
        
        completion = openai.Completion.create(
            prompt=summarizer_prompt,
            engine="text-davinci-003",
            max_tokens=cutoff,
        )

        generated_response = completion['choices'][0]['text'].strip()

        return generated_response
        

    def _trim_input(self, input: str, n_chunks: int = 5, reduction_multiplier: float = 0.75) -> str:
        whitespace_removal = input.replace(r' {2,}', "").strip()
        
        def split_string_into_chunks(s: str, n: int) -> List[str]:
            """Split a string into n equal chunks."""
            if n <= 0:
                raise ValueError("Number of chunks (n) must be greater than 0.")
            chunk_size = len(s) // n
            remainder = len(s) % n
            chunks = [s[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
            # Add the remaining characters to the last chunk
            if remainder:
                chunks[-1] += s[-remainder:]
            return chunks

        chunks = split_string_into_chunks(whitespace_removal, n_chunks)
        final = ""
        for chunk in chunks:
            final += self._summarize_chunk(chunk, int((len(whitespace_removal) // n_chunks) * reduction_multiplier))
        
        return final


    def _llm_reasoning_generation(self, input: str) -> str:
        """ Generates, given an input, a set of reasons from the context and input data. Should be outputted in provided format. """

        completion = openai.Completion.create(
            prompt=input,
            engine="text-davinci-003",
            max_tokens=500,
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
        <true | false>:<paragraph formatted reasoning>
        """

        completion = openai.Completion.create(
            prompt=final_classification_context,
            engine="text-davinci-003"
        )

        generated_response = completion['choices'][0]['text'].strip()
        decision, *reasons = generated_response.split(":")
        print(decision)
        return bool(decision), " ".join(reasons)
