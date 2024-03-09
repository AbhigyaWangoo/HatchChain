from llm.client import base
from query_engine.src.db import postgres_client
import json
import tqdm
from time import sleep
import os
from typing import List, Any

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

    def generate_prompt(
        self, job: str, resume: str, examples: str, max_length: int = 4096
    ) -> str:
        """Generates a prompt with the provided job and resume"""

        return f"""
            Given this resume: {resume}

            and this job description: {job}

            Would accept or reject the candidate for the provided
            job? Why or why not? If you are uncertain, you must decide
            whether to accept or reject based on the information you are provided.

            For example: {examples}
        """

    def append_to_file(self, fname: str, new_entries: List[Any]):
        """ Appends entries to an existing JSON file """

        if not os.path.exists(fname):
            open(fname, "w", encoding="utf8").close() # create if dne

        with open(fname, "r+", encoding="utf8") as fp:

            dataset = {}
            if os.path.getsize(fname) > 0:
                # If file is not empty.
                dataset = json.load(fp)
                dataset["dataset"] += new_entries
            else:
                dataset["dataset"] = new_entries

            fp.seek(0)

            json.dump(dataset, fp)

    def generate_dataset(self):
        """Generates a dataset in the output dataset filepath"""

        pgres_client = postgres_client.PostgresClient(self._job_id)
        resumes = pgres_client.read_candidates_from_job("", False, True)
        job = pgres_client.read_job(
            "/dev/null"
        )  # TODO maybe make this read job function just return in some cases

        examples = self.get_examples()
        for resume in tqdm.tqdm(resumes, desc="Generating dataset of resumes"):
            prompt = self.generate_prompt(job, resume, examples)

            response = "Response generation did not work"
            for i in range(MAX_RETRY):
                try:
                    response = self._client.query(prompt)
                    self.append_to_file(self._output_dateset_fpath, [response])
                    break
                except Exception as e:
                    if i == MAX_RETRY - 1:
                        print(f"Resume {resume[0]} couldn't be processed, moving on to next one")
                    else:
                        print(
                            f"Caught issue with llm client. Retrying resume...{e}"
                        )
                        sleep(1)
