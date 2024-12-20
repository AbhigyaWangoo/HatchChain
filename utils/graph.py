import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.dataset_cleaner import clean_indecisive_entires

DATA_DIR = "data/"
GRAPHS_DIR = "graphs/"

RELEVANCY_MATCH = r"Relevancy*"

EXPLAINABLE_CLASSIFICATION_REASONING_COL = "explainable_classification_reasoning"
EXPLAINABLE_CLASSIFICATION_COL = "explainable_classification"

NAMES_COL = "resume_path"
LINKS_COL = "linkedin links"
REASONING_COL = "reasoning"

REJECT='reject'
ACCEPT='accept'
TRUE="true"

class EvaluationResult:
    """
    A class to hold an evaluation result on one dataset for one role title for one recruiter.
    """

    def __init__(self, tp: int, fp: int, tn: int, fn: int) -> None:
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def precision(self) -> float:
        """Calculate and return precision"""
        return self.tp / (self.tp + self.fp)

    def recall(self) -> float:
        """Calculate and return recall"""
        return self.tp / (self.tp + self.fn)

    def f1(self) -> float:
        """Calculate and return F1 score"""
        return (2 * self.precision() * self.recall()) / (
            self.precision() + self.recall()
        )


class MetricGrapher:
    """A class to evaluation metrics for the explainable classifications"""

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.df = pd.read_csv(source_file)

    def generate_all_graphs(self):
        """Generates all individual graphs for evals"""

        patterns = {
            "Grammar.png": [r"Grammar and Readability:.*", int],
            "Accuracy.png": [r"Accuracy of explanation:.*", int],
            "Relevancy.png": [r"Relevancy:.*", int],
            "Detail.png": [r".*detailed.*", int],
        }

        for graph_name, pattern_and_type in patterns.items():
            self.generate_score_graph(
                pattern_and_type[1], pattern_and_type[0], graph_name
            )

    def eval_to_output(self, eval_csv: str, output_csv: str):
        """
        Converts the input csv into a readable output_csv for the f1 score reader function.
        """

        df = pd.read_csv(eval_csv)

        df = clean_indecisive_entires(eval_csv, "reasoning", EXPLAINABLE_CLASSIFICATION_COL, True)
       
        # RM links col. Not used for evaluation.
        df.drop(columns=[LINKS_COL], axis=1, inplace=True)
        df.drop(columns=["Unnamed*"], axis=1, inplace=True)
        # Also rename the reasoning col in the df to EXPLAINABLE_CLASSIFICATION_REASONING_COL
        df.rename(columns={"name": NAMES_COL}, inplace=True)
        
        # write same df back to file, but drop reasonings
        df.drop(columns=[REASONING_COL], axis=1, inplace=True)
        df.to_csv(output_csv)

    def generate_f1_scores(
        self, outputs_csv: str, ground_truth_csv: str
    ) -> EvaluationResult | None:
        """
        This function generates a map of f1 scores for a provided outputs dataset from a llm.

        outputs_csv: a csv file from some recruiter, which has a list of resumes and corresponding explainable classifications
        ground_truth_csv: a csv file for some role title which has 2 columns, a resume column, and a binary classification

        returns a set of scores
        """
        outputs_df = pd.read_csv(outputs_csv)
        ground_truth_df = pd.read_csv(ground_truth_csv)

        tn, tp, fn, fp = 0, 0, 0, 0

        try:
            classifications = outputs_df[EXPLAINABLE_CLASSIFICATION_COL].tolist()
            names = outputs_df[NAMES_COL].tolist()
            zipped_dict_outputs = {
                key: value for key, value in zip(names, classifications)
            }

            classifications = ground_truth_df[EXPLAINABLE_CLASSIFICATION_COL].tolist()
            names = ground_truth_df[NAMES_COL].tolist()
            zipped_dict_gt = {key: value for key, value in zip(names, classifications)}

            idx=0
            gt_lst_vals=list(zipped_dict_gt.values())
            for key in zipped_dict_outputs:
                classification = bool(zipped_dict_outputs[key])

                if key not in zipped_dict_gt:
                    gt_val = bool(gt_lst_vals[idx]) # index bullshit. In case the name matching is off. and it usually is.
                else:
                    gt_val = bool(zipped_dict_gt[key])

                if classification and gt_val:  # true positive
                    tp += 1
                elif classification and not gt_val:  # false positive
                    fp += 1
                elif not classification and gt_val:  # false negative
                    fn += 1
                elif not classification and not gt_val:  # true negative
                    tn += 1

                idx+=1

            return EvaluationResult(tp, fp, tn, fn)

        except KeyError as e:
            print(f"Keys dont exist, or something else: {e}")
            return None

    def generate_score_graph(
        self, type, pattern_str: str = RELEVANCY_MATCH, graph_name: str = "graph.png"
    ):
        """Generates a graph of the regex matched scores"""

        pattern = re.compile(pattern_str, re.IGNORECASE)

        key_column = "recruiter"
        relevant_columns = [
            column for column in self.df.columns if pattern.match(column)
        ]
        relevant_df = self.df[relevant_columns]
        relevant_df[relevant_columns] = relevant_df[relevant_columns].astype(int)

        numeric_columns = relevant_df.select_dtypes(include="number").columns
        relevant_numeric_columns = [
            col for col in relevant_columns if col in numeric_columns
        ]

        grouped_data_mean = self.df.groupby(key_column)[relevant_numeric_columns].mean()
        grouped_data_std = self.df.groupby(key_column)[relevant_numeric_columns].std()

        # Plot bar graph
        plt.figure(figsize=(15, 6))

        # Determine x positions for bars
        x_positions = np.arange(len(grouped_data_mean))

        bar_width = 0.2  # Adjust as needed

        for i, column in enumerate(relevant_numeric_columns):
            # Adjust x positions for each relevant column
            x_pos_adjusted = (
                x_positions + (i - (len(relevant_numeric_columns) - 1) / 2) * bar_width
            )

            # Plot bars for mean relevancy scores
            plt.bar(
                x_pos_adjusted,
                grouped_data_mean[column],
                width=bar_width,
                yerr=grouped_data_std[column],
                label=column,
            )

        title = graph_name.split(".")[0]
        plt.xlabel(key_column)
        plt.ylabel("Mean Relevancy Score")
        plt.title(f"{title} Scores by Recruiter")
        plt.xticks(x_positions, grouped_data_mean.index)
        plt.legend()
        output_path = os.path.join(os.path.join(DATA_DIR, GRAPHS_DIR), graph_name)
        plt.savefig(output_path)
