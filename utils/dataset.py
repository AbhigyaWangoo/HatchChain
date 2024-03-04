from llm.client import base
from query_engine.src.db import postgres_client
import csv


class DatasetGenerator:
    """
    A class to help generate datasets with the provided prompter
    """

    def __init__(
        self, output_dataset_filename: str, client: base.AbstractLLM, job_id: int
    ) -> None:
        super().__init__()
        self._output_dateset_fpath = output_dataset_filename
        self._client = client
        self._job_id = job_id

    def generate_dataset(self):
        """Generates a dataset in the output dataset filepath"""

        pgres_client = postgres_client.PostgresClient(self._job_id)
        resumes = pgres_client.read_candidates_from_job("", False, True)
        job = pgres_client.read_job(
            "/dev/null"
        )  # TODO maybe make this read job function just return in some cases

        with open(self._output_dateset_fpath, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Resume", "Explanation"])

            for resume in resumes:
                prompt = f"""
                    Given this resume: {resume}
                    
                    and this job description: {job}
                    
                    Would accept or reject the candidate for the provided
                    job? Why or why not? If you are uncertain, you must decide
                    whether to accept or reject based on the information you are provided.
                """

                response = self._client.query(prompt)
                csv_writer.writerow([resume, response])
