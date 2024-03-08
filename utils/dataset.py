from llm.client import base
from query_engine.src.db import postgres_client
import csv
import json
import tqdm
from time import sleep

EXAMPLES = "examples"
PROMPT = "prompt"
EXAMPLE = "example"
MAX_RETRY = 10


class DatasetGenerator:
    """
    A class to help generate datasets with the provided prompter
    """

    def __init__(
        self,
        output_dataset_filename: str,
        client: base.AbstractLLM,
        job_id: int,
        few_shot_examples_fname: str,
    ) -> None:
        super().__init__()
        self._output_dateset_fpath = output_dataset_filename
        self._client = client
        self._job_id = job_id
        self._few_shot_examples_fname = few_shot_examples_fname

    def get_examples(self) -> str:
        """
        Gets examples from the few shot examples file path and return a
        readable, concatenated string
        """

        with open(self._few_shot_examples_fname, "r", encoding="utf8") as fewshot_fp:
            few_shot_examples = json.load(fewshot_fp)
            examples = few_shot_examples[EXAMPLES]

            final_prompt = ""
            for example in examples:
                final_prompt += f"""
                    question: {example["prompt"]}

                    answer: {example["answer"]}
                """

            return final_prompt

    def generate_prompt(self, job: str, resume: str, examples: str, max_length: int=4096) -> str:
        """ Generates a prompt with the provided job and resume """

        if len(job) + len(examples) + len(resume) > max_length:
            

    def generate_dataset(self):
        """Generates a dataset in the output dataset filepath"""

        pgres_client = postgres_client.PostgresClient(self._job_id)
        resumes = pgres_client.read_candidates_from_job("", False, True)
        job = pgres_client.read_job(
            "/dev/null"
        )  # TODO maybe make this read job function just return in some cases

        output_json = []
        with open(
            self._output_dateset_fpath, "w", newline="", encoding="utf8"
        ) as csvfile:
            examples = self.get_examples()

            for resume in tqdm.tqdm(resumes, desc="Generating dataset of resumes"):
                prompt = f"""
                    Given this resume: {resume}

                    and this job description: {job}

                    Would accept or reject the candidate for the provided
                    job? Why or why not? If you are uncertain, you must decide
                    whether to accept or reject based on the information you are provided.

                    For example: {examples}
                """

                response = "Response generation did not work"
                for i in range(MAX_RETRY):
                    try:
                        response = self._client.query(prompt)
                        break
                    except Exception as e:

                        if i == MAX_RETRY - 1:
                            json.dump({"dataset": output_json}, csvfile)
                        else:
                            print(
                                f"Caught issue with llm client. Retrying resume...{e}"
                            )
                            sleep(2)

                output_json.append({"resume": resume, "explanation": response})

            json.dump({"dataset": output_json}, csvfile)
