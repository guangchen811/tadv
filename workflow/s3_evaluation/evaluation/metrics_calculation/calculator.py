import pandas as pd
from sklearn.metrics import roc_auc_score

from tadv.loader import FileLoader
from workflow.s3_evaluation.evaluation.metrics_calculation.abstract_calculator import AbstractMetricsCalculation


class MetricsCalculation(AbstractMetricsCalculation):
    def __init__(self):
        super().__init__()

    def calculate(self, downstream_task, script_output_dir):
        if downstream_task == "ml_inference_classification":
            result = self.calculate_classification_metrics(
                script_output_dir)
        elif downstream_task == "ml_inference_regression":
            result = self.calculate_regression_metrics(
                script_output_dir)
        elif downstream_task == "sql_query":
            result = self.calculate_sql_metrics(script_output_dir)
        elif downstream_task == "webpage_generation":
            result = self.calculate_webpage_metrics(
                script_output_dir)
        else:
            raise ValueError(f"downstream_task {downstream_task} is not supported")
        return result

    def calculate_classification_metrics(self, script_output_dir):
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        ground_truth_csv = FileLoader.load_csv(
            script_output_dir.parent.parent / "files_with_clean_new_data" / "ground_truth.csv")
        submission_on_corrupted_new_data = self._load_output_file_or_error(corrupted_new_data_path / "submission.csv")
        submission_on_clean_new_data = self._load_output_file_or_error(clean_new_data_path / "submission.csv")
        if isinstance(submission_on_corrupted_new_data, str) and submission_on_corrupted_new_data == "error":
            result_on_corrupted_new_data = "error"
        else:
            result_on_corrupted_new_data = self._calculate_auc(submission_on_corrupted_new_data, ground_truth_csv)

        if isinstance(submission_on_clean_new_data, str) and submission_on_clean_new_data == "error":
            result_on_clean_new_data = "error"
        else:
            result_on_clean_new_data = self._calculate_auc(submission_on_clean_new_data, ground_truth_csv)

        return {"result_on_corrupted_new_data": result_on_corrupted_new_data,
                "result_on_clean_new_data": result_on_clean_new_data}

    def calculate_regression_metrics(self, script_output_dir):
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        ground_truth_csv = FileLoader.load_csv(
            script_output_dir.parent.parent / "files_with_clean_new_data" / "ground_truth.csv")
        submission_on_corrupted_new_data = self._load_output_file_or_error(corrupted_new_data_path / "submission.csv")
        submission_on_clean_new_data = self._load_output_file_or_error(clean_new_data_path / "submission.csv")
        if isinstance(submission_on_corrupted_new_data, str) and submission_on_corrupted_new_data == "error":
            result_on_corrupted_new_data = "error"
        else:
            result_on_corrupted_new_data = self._calculate_mse(submission_on_corrupted_new_data, ground_truth_csv)

        if isinstance(submission_on_clean_new_data, str) and submission_on_clean_new_data == "error":
            result_on_clean_new_data = "error"
        else:
            result_on_clean_new_data = self._calculate_mse(submission_on_clean_new_data, ground_truth_csv)

        return {"result_on_corrupted_new_data": result_on_corrupted_new_data,
                "result_on_clean_new_data": result_on_clean_new_data}

    def calculate_sql_metrics(self, script_output_dir):
        # TODO: make it more representative
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        output_on_corrupted_new_data = self._load_output_file_or_error(corrupted_new_data_path / "output.csv")
        output_on_clean_new_data = self._load_output_file_or_error(clean_new_data_path / "output.csv")
        if isinstance(output_on_corrupted_new_data, str) and output_on_corrupted_new_data == "error":
            result_on_corrupted_new_data = "error"
        else:
            result_on_corrupted_new_data = "success"

        if isinstance(output_on_clean_new_data, str) and output_on_clean_new_data == "error":
            result_on_clean_new_data = "error"
        else:
            result_on_clean_new_data = "success"

        return {"result_on_corrupted_new_data": result_on_corrupted_new_data,
                "result_on_clean_new_data": result_on_clean_new_data}

    def calculate_webpage_metrics(self, script_output_dir):
        # TODO: make it more representative
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        file_suffix_set_on_corrupted_new_data = set(
            [file.suffix for file in corrupted_new_data_path.iterdir() if file.is_file()])
        file_suffix_set_on_clean_new_data = set(
            [file.suffix for file in clean_new_data_path.iterdir() if file.is_file()])
        if ".html" in file_suffix_set_on_corrupted_new_data:
            result_on_corrupted_new_data = "success"
        else:
            result_on_corrupted_new_data = "error"

        if ".html" in file_suffix_set_on_clean_new_data:
            result_on_clean_new_data = "success"
        else:
            result_on_clean_new_data = "error"

        return {"result_on_corrupted_new_data": result_on_corrupted_new_data,
                "result_on_clean_new_data": result_on_clean_new_data}

    @staticmethod
    def _load_output_file_or_error(file_path):
        if file_path.exists():
            return FileLoader.load_csv(file_path)
        else:
            return "error"

    @staticmethod
    def _calculate_auc(submission_on_corrupted_new_data, ground_truth_csv):
        # Extract unique class labels
        classes = ground_truth_csv.iloc[:, -1].unique()
        str_int_mapping = {cls: i for i, cls in enumerate(classes)}

        # Map classes to integers
        y_true = ground_truth_csv.iloc[:, -1].map(str_int_mapping).copy()
        y_pred = submission_on_corrupted_new_data.iloc[:, -1].map(str_int_mapping).copy()

        # One-hot encode for multi-class
        if len(classes) > 2:
            y_true = pd.get_dummies(y_true).values  # Convert to NumPy array
            y_pred_proba = pd.get_dummies(y_pred).values  # Convert to NumPy array
            auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        else:
            auc = roc_auc_score(y_true, y_pred)

        return auc

    @staticmethod
    def _calculate_mse(submission_on_corrupted_new_data, ground_truth_csv):
        y_true = ground_truth_csv.iloc[:, -1]
        y_pred = submission_on_corrupted_new_data.iloc[:, -1]
        mse = ((y_true - y_pred) ** 2).mean()
        return mse
