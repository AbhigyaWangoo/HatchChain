import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np


GRAPHS_DIR="graphs/"

RELEVANCY_MATCH=r"Relevancy*"

class MetricGrapher():
    """A class to evaluation metrics for the explainable classifications"""
    def __init__(self, source_file: str) -> None:
        self.source_file=source_file
        self.df = pd.read_csv(source_file)

    def generate_relevancy_score_graph(self, pattern_str: str=RELEVANCY_MATCH, graph_name: str="graph.png"):
        """Generates a graph of the relevancy scores"""

        pattern = re.compile(pattern_str)

        key_column="recruiter"
        relevant_columns = [column for column in self.df.columns if pattern.match(column)]
        numeric_columns = self.df.select_dtypes(include='number').columns
        relevant_numeric_columns = [col for col in relevant_columns if col in numeric_columns]

        grouped_data_mean = self.df.groupby(key_column)[relevant_numeric_columns].mean()
        grouped_data_std = self.df.groupby(key_column)[relevant_numeric_columns].std()

        # Plot bar graph
        plt.figure(figsize=(15, 6))

        # Determine x positions for bars
        x_positions = np.arange(len(grouped_data_mean))

        bar_width = 0.2  # Adjust as needed

        for i, column in enumerate(relevant_numeric_columns):
            # Adjust x positions for each relevant column
            x_pos_adjusted = x_positions + (i - (len(relevant_numeric_columns) - 1) / 2) * bar_width

            # Plot bars for mean relevancy scores
            plt.bar(x_pos_adjusted, grouped_data_mean[column], width=bar_width, yerr=grouped_data_std[column], label=column)

        plt.xlabel(key_column)
        plt.ylabel('Mean Relevancy Score')
        plt.title('Relevancy Scores by Recruiter')
        plt.xticks(x_positions, grouped_data_mean.index)
        plt.legend()
        plt.savefig(graph_name)
