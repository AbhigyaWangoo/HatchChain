# from dspy.teleprompt import BootstrapFewShotWithRandomSearch
# from dspy.evaluate import Evaluate
# import dspy
# from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
# from classifier.decisiontree import ExplainableTreeClassifier
# import openai
import os
import pickle
from typing import Dict, Set, List

from utils.utils import read_from, TESTDIR, LABELS

# client = openai.OpenAI()


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
            lbls = read_from(dir_path + file)
            data = read_from(dir_path + base + ".txt")[0].strip()

            for lbl_line in LABELS:
                try:
                    classifier = ExplainableTreeClassifier(
                        ["Experiences", "Skills"], lbl_line
                    )
                    decision, reason = classifier.classify(data)

                    print(
                        f"FOR CANDIDATE {file}, we {decision} for the position of {lbl_line} because {reason}"
                    )
                    if (lbl_line in lbls and decision) or (
                        lbl_line not in lbls and not decision
                    ):
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
    print("Hello world!")
