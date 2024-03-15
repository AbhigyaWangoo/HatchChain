from llm.client import base, mistral
from query_engine.src.db import postgres_client
import json
import tqdm
from time import sleep
import os
from typing import List, Any, Dict, Tuple
from typeguard import typechecked

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
        self._mistral_summarizor = mistral.MistralLLMClient()

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

        new_job = self._mistral_summarizor.query(
            f"Summarize the following job into 5 sentences. Preserve key information, and do not change any narrative: {job}"
        )

        total_length = len(new_job) + len(resume) + len(examples)
        if total_length > max_length:
            resume = self._mistral_summarizor.query(
                f"Summarize the following resume into 5 sentences. Preserve key information, and do not change any narrative: {resume}"
            )

        return f"""
            Given this resume: {resume}

            and this job description: {new_job}

            Would accept or reject the candidate for the provided
            job? Why or why not? If you are uncertain, you must decide
            whether to accept or reject based on the information you are provided.

            For example: {examples}
        """

    def append_to_file(self, fname: str, new_entries: List[Any]):
        """Appends entries to an existing JSON file"""

        if not os.path.exists(fname):
            open(fname, "w", encoding="utf8").close()  # create if dne

        with open(fname, "r+", encoding="utf8") as fp:

            dataset = {}
            if os.path.getsize(fname) > 0:
                # If file is not empty.
                dataset = json.load(fp)
                dataset["dataset"] += new_entries
            else:
                dataset["dataset"] = new_entries

            fp.seek(0)
            fp.truncate(0)

            json.dump(dataset, fp)

    def _clean_dict(
        self, data: Dict[str, Any], entries_to_rm: List[str]
    ) -> Dict[str, Any]:
        """Removes the provided entries from the dict"""

        for x in entries_to_rm:
            if x in data:
                del data[x]

        return data

    @typechecked
    def _clean_job(self, job: Dict[Any, Any]) -> str:
        """Cleans up a job and returns a string."""
        entries_to_rm = [
            "id",
            "recruiter_id",
            "rank_up_to_date",
            "ideal_custom_score",
            "classifier_data",
        ]

        return str(self._clean_dict(job, entries_to_rm))

    @typechecked
    def _clean_resume(self, resume: Tuple[Any, Any]) -> str:
        """Cleans up a resume and returns a string."""
        data = resume[1]
        entries_to_rm = ["email", "phone", "links"]

        return str(self._clean_dict(data, entries_to_rm))

    def generate_dataset(self):
        """Generates a dataset in the output dataset filepath"""

        pgres_client = postgres_client.PostgresClient(self._job_id)
        resumes = pgres_client.read_candidates_from_job(
            "", False, True
        )  # TODO Also need to add *args for selective column return
        job = pgres_client.read_job(
            "/dev/null"
        )  # TODO maybe make this read job function just return in some cases. Also need to add *args for selective column return
        job = self._clean_job(job)

        examples = self.get_examples()
        found = set()
        with open(self._output_dateset_fpath, "r", encoding="utf8") as fp:
            try:
                dataset = json.load(fp)
                for resume in dataset["dataset"]:
                    found.add(resume["resume"][0])
            except Exception as e:
                print(f"Error in reading dataset {e}")

        print(len(found))
        for resume in tqdm.tqdm(resumes, desc="Generating dataset of resumes"):
            if resume[0] in found:
                continue

            resume_str = self._clean_resume(resume)
            prompt = self.generate_prompt(job, resume_str, str(examples))

            response = "Response generation did not work"
            for i in range(MAX_RETRY):
                try:
                    response = self._client.query(prompt)
                    if len(response) < 10:
                        print("TOKEN LIMIT EXCEEDED")
                        exit(1)

                    self.append_to_file(
                        self._output_dateset_fpath,
                        [{"resume": resume, "explanation": response}],
                    )
                    break
                except Exception as e:
                    if i == MAX_RETRY - 1:
                        print(
                            f"Resume {resume[0]} couldn't be processed, moving on to next one"
                        )
                    else:
                        print(f"Caught issue with llm client. Retrying resume...{e}")
                        sleep(1)
