from . import base as prompter
from llm.client import base as llm
import dspy
import os
from dspy.teleprompt import BootstrapFewShot
from dspy import Example
from typing import Tuple, List, Callable
import json
from utils.utils import generate_random_integer

JOB_CONTEXT = "heuristic"
OUTPUT = "navigation"
INPUT_RESUME = "input resume"

DEFAULT_TEST_SPLIT = 0.8
THRESHOLD = 1.0
CATEGORY = "category"
N = 5


class DSPyPrompter(prompter.Prompter):
    """
    A class for DSPy prompting. Based off paper
    """

    def __init__(
        self,
        client: llm.AbstractLLM,
        dataset: str,
        prompt_crafter: Callable[[str, str], str],
        train_test_split: float = DEFAULT_TEST_SPLIT,
    ) -> None:
        self._client = client
        self._train, self.test = self.load_dataset(
            dataset, prompt_crafter, train_test_split
        )
        self._classify = dspy.ChainOfThought("question -> answer", n=N)
        self.train()

    def prompt(self, prompt: str) -> str:
        """
        The abstract method for a prompter to execute a prompt
        """
        # 2. call with input
        response = self._classify(question=prompt)

        # 3. Access output
        return response.completions.answer

    def train(self):
        """
        Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
        The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
        """

        config = dict(
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            # num_candidate_programs=10,
            # num_threads=4,
        )

        teleprompter = BootstrapFewShot(
            metric=self.validate_context_and_answer, **config
        )
        optimized_program = teleprompter.compile(self._classify, trainset=self._train)

        self._classify = optimized_program

    def load_dataset(
        self,
        dataset: str,
        prompt_crafter: Callable[[str, str, str], str],
        train_test_split: float = DEFAULT_TEST_SPLIT,
    ) -> Tuple[List[Example], List[Example]]:
        """
        Returns a (train, test) dataset.
        """
        train, test = [], []

        if not os.path.exists(dataset):
            print(
                f"Path {dataset} does not exist, please check with a different dataset"
            )

        with open(dataset, "r", encoding="utf8") as fp:
            corpus = json.load(fp)
            dataset_size = len(corpus)
            num_train = int(dataset_size * train_test_split)

            for item in corpus:
                try:
                    context = item[JOB_CONTEXT]
                    example_output = item[OUTPUT]
                    input_resume = item[INPUT_RESUME]
                    category = item[CATEGORY]
                except KeyError as k:
                    print(f"Dataset {dataset} was invalid, error: {k}")
                    return train, test

                question = prompt_crafter(context, input_resume, category)
                answer = example_output

                example = Example(question=question, answer=answer)
                res = generate_random_integer(dataset_size)

                if res >= 1 and res <= num_train:
                    train.append(example)
                else:
                    test.append(example)

        return train, test

    def validate_context_and_answer(self, example: Example, pred: Example, trace=None):
        """
        --------EVALUATION METRICS-----
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

        score = 0.0

        # -----Keyword retention-----
        # 1. turn job into set of words
        job_description = set(pred.question.split(" "))

        # 2. intersect the job set with the answer set
        reasoning = set(pred.answer.split(" "))

        # 3. divide by Number of keywords in original passage. Assumming a 60% length impact size.
        job_length = len(job_description.intersection(reasoning)) / len(pred.question)

        # add to total numerator and denominator
        score += job_length

        # -----Classification-----
        # 1. get num_accpet and num_reject from example
        reject_ct_example = example.answer.lower().count("reject")
        accept_ct_example = example.answer.lower().count("accept")

        # 2. get num_accpet and num_reject from pred
        reject_ct_pred = pred.answer.lower().count("reject")
        accept_ct_pred = pred.answer.lower().count("accept")

        # 3. both should have either positive or negative diff
        if (
            reject_ct_example > accept_ct_example and reject_ct_pred > accept_ct_pred
        ) or (
            reject_ct_example < accept_ct_example and reject_ct_pred < accept_ct_pred
        ):
            score += 1.0

        if trace is None:  # if we're doing evaluation or optimization
            return score
        else:  # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
            return score > THRESHOLD


class DSpyCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
        self.turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=250)
        dspy.settings.configure(lm=self.turbo)

    def forward(self, question):
        return self.prog(question=question)
