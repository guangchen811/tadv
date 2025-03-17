import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import trim_mean, gmean
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, recall_score, f1_score
)

from tadv.utils import get_current_folder


class RelevantColumnDetectionMetric:
    """
    A class to compute metrics for multi-label classification tasks,
    such as accuracy, Hamming loss, precision, recall, and F1 score.
    """

    def __init__(self, average: str = 'macro'):
        """
        Initialize the metric class.

        Parameters:
        - average (str): Averaging method for precision, recall, and F1.
                         Options: 'micro', 'macro', 'weighted', 'samples'
                         Default: 'macro'.
        """
        self.average = average  # Averaging strategy

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the performance of multi-label classification.

        Parameters:
        - y_true (array-like): Ground truth binary labels (shape: N x L).
        - y_pred (array-like): Predicted binary labels (shape: N x L).

        Returns:
        - metrics_with_stats (dict): A dictionary containing metrics with their means and standard deviations.
        """
        # Convert to NumPy arrays

        statistics = self.statistics_calculation(y_pred,
                                                 y_true)
        accuracy_list, f1_list, hamming_list, precision_list, recall_list = statistics["accuracy_list"], statistics[
            "f1_list"], statistics["hamming_list"], statistics["precision_list"], statistics["recall_list"]

        # Compute mean and standard deviation for each metric
        metrics_with_stats = {
            "Accuracy": f"{np.mean(accuracy_list):.4f} ± {np.var(accuracy_list):.4f}",
            "Hamming Loss": f"{np.mean(hamming_list):.4f} ± {np.var(hamming_list):.4f}",
            f"Precision ({self.average})": f"{np.mean(precision_list):.4f} ± {np.var(precision_list):.4f}",
            f"Recall ({self.average})": f"{np.mean(recall_list):.4f} ± {np.var(recall_list):.4f}",
            f"F1 Score ({self.average})": f"{np.mean(f1_list):.4f} ± {np.var(f1_list):.4f}"
        }
        return metrics_with_stats

    def statistics_calculation(self, y_pred, y_true):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracy_list, hamming_list = [], []
        precision_list, recall_list, f1_list = [], [], []
        for i in range(y_true.shape[0]):
            accuracy_list.append(accuracy_score(y_true[i, :], y_pred[i, :]))
            hamming_list.append(hamming_loss(y_true[i, :], y_pred[i, :]))
            precision_list.append(precision_score(y_true[i, :], y_pred[i, :], zero_division=0))
            recall_list.append(recall_score(y_true[i, :], y_pred[i, :], zero_division=0))
            f1_list.append(f1_score(y_true[i, :], y_pred[i, :], zero_division=0))

        # Compute aggregated metrics for F1 scores
        median_f1 = np.median(f1_list)
        trimmed_mean_f1 = trim_mean(f1_list, 0.1)  # Trim 10% from both ends
        geometric_mean_f1 = gmean(np.maximum(f1_list, 1e-6))  # Avoid zero values for geometric mean
        harmonic_mean_f1 = len(f1_list) / np.sum(1 / np.maximum(f1_list, 1e-6))  # Avoid division by zero
        statistics = {
            "accuracy_list": accuracy_list,
            "f1_list": f1_list,
            "hamming_list": hamming_list,
            "precision_list": precision_list,
            "recall_list": recall_list,
            "median_f1": median_f1,
            "trimmed_mean_f1": trimmed_mean_f1,
            "geometric_mean_f1": geometric_mean_f1,
            "harmonic_mean_f1": harmonic_mean_f1,
        }
        return statistics

    def binary_vectorize(self, all_columns, ground_truth, relevant_columns):
        ground_truth_vector = [1 if col in ground_truth else 0 for col in all_columns]
        relevant_columns_vector = [1 if col in relevant_columns else 0 for col in all_columns]
        return ground_truth_vector, relevant_columns_vector

