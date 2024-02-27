from . import base as prompter
from llm.client import base as llm
import dspy
import os
from dspy import Example
from typing import Tuple, List, Callable
import json
from utils.utils import generate_random_integer

JOB_CONTEXT="heuristic"
OUTPUT="navigation"
INPUT_RESUME="input resume"

DEFAULT_TEST_SPLIT=0.8

class DSPyPrompter(prompter.Prompter):
    """
    A class for DSPy prompting. Based off paper
    """

    def __init__(self, client: llm.AbstractLLM, dataset: str, prompt_crafter: Callable[[str, str], str], train_test_split: float = DEFAULT_TEST_SPLIT) -> None:
        self._client = client
        self.train, self.test = self.load_dataset(dataset, prompt_crafter, train_test_split)

    def prompt(self, prompt: str) -> str:
        """
        The abstract method for a prompter to execute a prompt
        """
        pass

    def load_dataset(self, dataset: str, prompt_crafter: Callable[[str, str], str], train_test_split: float = DEFAULT_TEST_SPLIT) -> Tuple[List[Example], List[Example]]:
        """
        Returns a (train, test) dataset.
        """
        train, test = [], []

        if not os.path.exists(dataset):
            print(f"Path {dataset} does not exist, please check with a different dataset")

        with open(dataset, "r", encoding="utf8") as fp:
            corpus = json.load(fp)
            dataset_size = len(corpus)
            num_train = int(dataset_size * train_test_split)

            for item in corpus:
                try:
                    context = item[JOB_CONTEXT]
                    example_output = item[OUTPUT]
                    input_resume = item[INPUT_RESUME]
                except KeyError as k:
                    print(f"Dataset {dataset} was invalid, error: {k}")
                    return train, test

                question=prompt_crafter(context, input_resume)
                answer=example_output

                example = Example(question=question, answer=answer)
                res = generate_random_integer(dataset_size)

                if res >= 1 and res <= num_train:
                    train.append(example)
                else:
                    test.append(example)

        return train, test

    def validate_context_and_answer(self, example: Example, pred: Example, trace=None):
        """
        Relevancy
            Relevance towards job description: For each component of the job description, did the reasoning mention or reference it? Tally the total number of references.
            Hallucination: Tally from the job description
            Relevance towards resumes: For each hyperparameter, was there a relevant reference to that candidate’s resume? Tally the total number of references.
        Hallucination: Tally from the resume
            On a scale of 1-5, how detailed were the explanations with respect to the reusme?
        Classification Accuracy
            (TP + TN) / Total classifications
        Grammar and Readability
            On a scale of 1-5, how correct was the grammar in the explanation?
            On a scale of 1-5, how readable was the language used in the explanation?
            On a scale of 1-5, how properly did the language build up over the course of the explanation? (Clear transitions between hyperparameters, clear final conclusion drawing from hyperparameters?)
        Safety
            Did the LLM leak any non-understandable information about the resume or jobdescription, or about Hatch’s internals?
        Classification Efficiency
            In the case of an acceptance, on a scale of 1-5, how indicative was the explanation of the candidate’s ranking?
            In the case of an rejection, on a scale of 1-5, how indicative was the explanation of the candidate’s ranking?
            
        A DSPy metric is just a function in Python that takes example (e.g., from your training or dev set) and the 
        output pred from your DSPy program, and outputs a float (or int or bool) score.
        
        Defining a good metric is an iterative process, so doing some initial evaluations and looking at your data and your outputs are key.
        """

        # check the predicted answer comes from one of the retrieved contexts
        context_match = any((pred.answer.lower() in c) for c in pred.context)
        # job similarity
        # 1. turn job into set of words
        # 2. intersect the job set with the answer set
        # 3. divide by ??

        # Resume similarity
        # 1. turn resume into set of words
        # 2. intersect the resume set with the answer set
        # 3. divide by ??

        if trace is None: # if we're doing evaluation or optimization
            return (answer_match + context_match) / 2.0
        else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
            return answer_match and context_match


class DSpyCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
        self.turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=250)
        dspy.settings.configure(lm=self.turbo)

    def forward(self, question):
        return self.prog(question=question)
