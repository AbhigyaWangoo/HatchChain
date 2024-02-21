from query_engine.src.db import postgres_client
from classifier.decisiontree import ExplainableTreeClassifier
import openai
import os
import pickle
from typing import Dict, Set, List

from llm import runpod
from nltk.tokenize import word_tokenize
from utils.utils import read_from, TESTDIR, LABELS
from memgpt import MemGPT

client = openai.OpenAI()


def run_binclassifier(dir_path: str, n: int = -1) -> Dict[str, str]:
    """ 
    Returns a dict of {filename: label}
    """

    file_to_lbl = {}
    files = os.listdir(dir_path)
    files.sort()

    total = 0.0
    s = 0.0

    y_pred = []
    for file in files:
        base, ext = os.path.splitext(file)

        if ext == ".lab":
            lbls = read_from(dir_path+file)
            data = read_from(dir_path+base+".txt")[0].strip()

            for lbl_line in LABELS:
                try:
                    classifier = ExplainableTreeClassifier(
                        ["Experiences", "Skills"], lbl_line)
                    decision, reason = classifier.classify(data)

                    print(
                        f"FOR CANDIDATE {file}, we {decision} for the position of {lbl_line} because {reason}")
                    if (lbl_line in lbls and decision) or (lbl_line not in lbls and not decision):
                        s += 1.0
                    total += 1.0

                    print(f"Running accuracy= {s/total}\r")

                    if n > 0:
                        n -= 1
                    elif n <= 0:
                        break
                except openai.BadRequestError:
                    print("CONTEXT LENGTH EXCEEDED, CONTINUING")

            data = ""
            lbl = ""

    print(f"FINAL accuracy= {s/total}\r")
    return file_to_lbl

if __name__ == "__main__":
    # runpod_client = runpod.RunPodClient()
    # print(runpod_client.query("Hello, how are you?", False))
    tree = ExplainableTreeClassifier(["Experiences", "skills"],
                                     "Database_Administrator", "local.json")
    # similar_dict = tree.get_k_similar(152, 2186, 3, True)
    # for key in similar_dict:
    #     print(key)
    #     print(similar_dict[key])
    res = tree.tiebreak(2186, 152, [])