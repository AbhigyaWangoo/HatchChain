import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import numpy as np

DATA_DIR = "data/"
GRAPHS_DIR = "graphs/"

RELEVANCY_MATCH = r"Relevancy*"


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
            "Detail.png": [r".*detailed.*", int]
        }

        for graph_name, pattern_and_type in patterns.items():
            self.generate_score_graph(
                pattern_and_type[1], pattern_and_type[0], graph_name
            )

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
