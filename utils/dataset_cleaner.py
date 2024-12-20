import os
import json
import csv
from query_engine.src.db import postgres_client
from llm.client import gpt
import tqdm
import pandas as pd

DIRSRC = "reports"

RELEVANCY_COL_0 = "Relevancy: For each component of the job description, did the reasoning mention or reference it? Tally the total number of references."
RELEVANCY_COL_1 = "Relevancy: Did the reasoning mention an information that was incorrect from the job description? Tally the total number of references."
RELEVANCY_COL_2 = "Relevancy: Were there a relevant reference to that candidate’s resume? Tally the total number of references."
RELEVANCY_COL_3 = "Relevancy: Did the reasoning mention an information that was incorrect from the resume? Tally the total number of references."

EVALS_FILE = "data/evals/eval_responses.csv"
NUMERICAL_SYSPROMPT = "utils/sysprompts/numerical_prompt.txt"
SENTIMENT_ANALYSIS_SYSPROMPT = "utils/sysprompts/reasoning_sentiment.txt"

REJECT="reject"
ACCEPT="accept"

def clean_classifier_results(eval_csv: str = EVALS_FILE, ofile: str = "output.json"):
    """
    Reads the evaluations from a csv file, and does the following:

    1. Converts all qualitative results into quantitative results
    2. Writes all of them to a new file

    """

    if not os.path.exists(eval_csv):
        print("Eval file does not exist. Exiting...")
        return

    df = pd.read_csv(eval_csv)
    unclean_columns = [
        RELEVANCY_COL_0,
        RELEVANCY_COL_1,
        RELEVANCY_COL_2,
        RELEVANCY_COL_3,
    ]

    unique_set = set()
    for column in unclean_columns:
        unique_set.update(df[column].unique())

    unique_set.remove("o")
    unclean_to_clean = {"o": 0}

    sysprompt = None
    with open(NUMERICAL_SYSPROMPT, "r", encoding="utf8") as fp:
        sysprompt = fp.read()

    client = gpt.GPTClient()
    for val in unique_set:

        if not val.isnumeric():

            res = client.query(val, sys_prompt=sysprompt, is_json=True)
            unclean_to_clean[val] = res["rating"]

    with open(ofile, "w", encoding="utf8") as fp:
        json.dump(unclean_to_clean, fp)

    print(unclean_to_clean)


def process_directory(directory_path):
    # Iterate through each file in the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)

                try:
                    with open(file_path, "r", encoding="utf8") as fp:
                        dataset = json.load(fp)
                        dataset_data = dataset["dataset"]

                        for resume_explanation in dataset_data:
                            if resume_explanation["resume"][1].get("links", None):
                                resume_explanation["resume"] = resume_explanation[
                                    "resume"
                                ][1]["links"]
                            elif resume_explanation["resume"][1].get("name", None):
                                resume_explanation["resume"] = resume_explanation[
                                    "resume"
                                ][1]["name"]

                        # print(dataset)# Write the modified JSON
                        ofilename = file_path.split("/")[-1]
                        with open(f"clean-{ofilename}", "w", encoding="utf8") as fp:
                            json.dump(dataset, fp)

                except Exception as e:
                    print(f"error writing customer back: {e}")

def clean_indecisive_entires(dataset_csv: str, reasoning_col: str, new_classification_col: str, write_back: bool=False) -> pd.DataFrame:
    """
    Takes all indecisive responses and forces them to have a binary classificartion.
    """
    df = pd.read_csv(dataset_csv)
    col = df[reasoning_col].to_list()

    if new_classification_col in df:
        print(f"Column {new_classification_col} already exists. exiting.")
        return df

    df[new_classification_col] = False
    with open(SENTIMENT_ANALYSIS_SYSPROMPT, "r", encoding="utf8") as fp:
        sysprompt=fp.read()
        client = gpt.GPTClient()

        n_unknowns=0
        for idx, reasoning in enumerate(col):
            binclassify=False

            if REJECT in reasoning.lower() and reasoning.count(REJECT)>reasoning.count(ACCEPT):
                binclassify=False
            elif ACCEPT in reasoning.lower() and reasoning.count(ACCEPT)>reasoning.count(REJECT):
                binclassify=True
            else:
                n_unknowns+=1
                response = client.query(sys_prompt=sysprompt, prompt=reasoning)
                binclassify ="true" in response.lower()

            df.at[idx, new_classification_col] = binclassify

        print(f"Num unknowns for {dataset_csv}: {n_unknowns}")

    if write_back:
        df.to_csv(dataset_csv)

    return df


def convert_to_csv(directory_path: str = DIRSRC):
    """Converts all the provided json files into csv files"""
    name_to_jobid = {"ml": 148, "frontend": 135, "backend": 188, "pm": 172}

    headers = ["name", "linkedin links", "reasoning"]
    for root, dirs, files in tqdm.tqdm(
        os.walk(directory_path), desc="Walking through directories"
    ):
        for file_name in tqdm.tqdm(files, desc="Walking through json files"):
            if file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                basename = file_name.split(".")[0]
                csv_file = os.path.join(root, f"{basename}.csv")

                type = basename.split("-")[1]
                job_id = name_to_jobid[type]
                client = postgres_client.PostgresClient(job_id)

                try:
                    with open(file_path, "r", encoding="utf8") as fp:
                        dataset = json.load(fp)
                        dataset_data = dataset["dataset"]

                        with open(
                            csv_file, "w", newline="", encoding="utf8"
                        ) as csvfile:
                            dict_writer = csv.DictWriter(
                                csvfile, fieldnames=headers, delimiter=","
                            )
                            dict_writer.writeheader()

                            for resume_explanation in tqdm.tqdm(
                                dataset_data,
                                desc=f"Generating dataset for file {basename}",
                            ):
                                resume = resume_explanation["resume"]
                                explanation = resume_explanation["explanation"]

                                name = resume[1].get("name", None)
                                links = resume[1].get("links", None)
                                if links is None:
                                    links = client.read_candidate(
                                        resume[0], "resume_data"
                                    )[0]["links"]
                                links = "______".join(links)

                                csv_row = {
                                    f"{headers[0]}": name,
                                    f"{headers[1]}": links,
                                    f"{headers[2]}": explanation,
                                }
                                dict_writer.writerow(csv_row)

                except Exception as e:
                    print(f"error converting to csv: {e}")
