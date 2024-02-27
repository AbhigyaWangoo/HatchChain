# from dspy.teleprompt import BootstrapFewShotWithRandomSearch
# from dspy.evaluate import Evaluate
# import dspy
# from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from classifier.decisiontree import ExplainableTreeClassifier
import openai
import os
import pickle
from typing import Dict, Set, List

from utils.utils import read_from, TESTDIR, LABELS

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


# class CoT(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         self.prog = dspy.ChainOfThought("question -> answer")

#     def forward(self, question):
#         return self.prog(question=question)


# def validate_context_and_answer(example, pred, trace=None):
# """A metric function for the DSpy module"""

# check the gold label and the predicted answer are the same
# answer_match = example.sentiment == pred.sentiment
# return answer_match

# check the predicted answer comes from one of the retrieved contexts
# context_match = any((pred.answer.lower() in c) for c in pred.context)

# if trace is None: # if we're doing evaluation or optimization
#     return (answer_match + context_match) / 2.0
# else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
#     return answer_match and context_match


def track_heuristics(node, res, input_str):
    import json

    with open("tracker.json", "r", encoding="utf8") as fp:
        data = json.load(fp)
        data.append(
            {
                "heuristic": node.heuristic,
                "navigation": res,
                "input resume": input_str,
            }
        )
        with open("tracker.json", "w", encoding="utf8") as fp:
            json.dump(data, fp)


if __name__ == "__main__":
    # turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=250)
    # dspy.settings.configure(lm=turbo)

    # gms8k = GSM8K()
    # trainset, devset = gms8k.train, gms8k.dev
    # teleprompter = BootstrapFewShotWithRandomSearch(
    #     metric=validate_context_and_answer,
    #     max_bootstrapped_demos=8,
    #     max_labeled_demos=8,
    # )
    import llm.prompt.dspy as dspyclient
    import llm.client.gpt as gptclient

    dt = ExplainableTreeClassifier([], "", None, "data/classifiers/166.json")
    dspy_client = dspyclient.DSPyPrompter(gptclient.GPTClient(),"dataset_10.json", dt.get_navigation_string)
    
