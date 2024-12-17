import numpy as np
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, recall_score, f1_score
)


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
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Initialize list to collect individual metrics for each label
        accuracy_list, hamming_list = [], []
        precision_list, recall_list, f1_list = [], [], []

        # Evaluate metrics label by label
        for i in range(y_true.shape[1]):  # Iterate over each column (label)
            accuracy_list.append(accuracy_score(y_true[:, i], y_pred[:, i]))
            hamming_list.append(hamming_loss(y_true[:, i], y_pred[:, i]))
            precision_list.append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
            recall_list.append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
            f1_list.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))

        # Compute mean and standard deviation for each metric
        metrics_with_stats = {
            "Accuracy": f"{np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}",
            "Hamming Loss": f"{np.mean(hamming_list):.4f} ± {np.std(hamming_list):.4f}",
            f"Precision ({self.average})": f"{np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}",
            f"Recall ({self.average})": f"{np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}",
            f"F1 Score ({self.average})": f"{np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}"
        }
        return metrics_with_stats

    def binary_vectorize(self, all_columns, ground_truth, relevant_columns):
        ground_truth_vector = [1 if col in ground_truth else 0 for col in all_columns]
        relevant_columns_vector = [1 if col in relevant_columns else 0 for col in all_columns]

        return ground_truth_vector, relevant_columns_vector


# Example usage
if __name__ == "__main__":
    # Example ground truth and predictions
    y_true = [[1, 0, 1]]
    y_pred = [[1, 0, 1]]

    # Instantiate the metric class
    metric_evaluator = RelevantColumnDetectionMetric(average='macro')
    results = metric_evaluator.evaluate(y_true, y_pred)

    # Print the results
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.4f}")
