from sklearn.metrics import roc_auc_score

from scripts.python.evaluation.metrics_calculation.abstract_calculator import AbstractMetricsCalculation
from tadv.loader import FileLoader


class MetricsCalculation(AbstractMetricsCalculation):
    def __init__(self):
        super().__init__()

    def calculate(self, dataset_name, downstream_task, processed_data_label, script_output_dir):
        if downstream_task == "ml_inference_classification":
            result = self.calculate_classification_metrics(dataset_name, downstream_task, processed_data_label,
                                                           script_output_dir)
        elif downstream_task == "ml_inference_regression":
            result = self.calculate_regression_metrics(dataset_name, downstream_task, processed_data_label,
                                                       script_output_dir)
        elif downstream_task == "sql_query":
            result = self.calculate_sql_metrics(dataset_name, downstream_task, processed_data_label, script_output_dir)
        elif downstream_task == "webpage_generation":
            result = self.calculate_webpage_metrics(dataset_name, downstream_task, processed_data_label,
                                                    script_output_dir)
        else:
            raise ValueError(f"downstream_task {downstream_task} is not supported")
        return result

    def calculate_classification_metrics(self, dataset_name, downstream_task, processed_data_label, script_output_dir):
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        ground_truth_csv = FileLoader.load_csv(
            script_output_dir.parent.parent / "files_with_clean_new_data" / "ground_truth.csv")
        submission_on_corrupted_new_data = self._load_csv_or_error(corrupted_new_data_path / "submission.csv")
        submission_on_clean_new_data = self._load_csv_or_error(clean_new_data_path / "submission.csv")
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

    def calculate_regression_metrics(self, dataset_name, downstream_task, processed_data_label, script_output_dir):
        raise NotImplementedError

    def calculate_sql_metrics(self, dataset_name, downstream_task, processed_data_label, script_output_dir):
        raise NotImplementedError

    def calculate_webpage_metrics(self, dataset_name, downstream_task, processed_data_label, script_output_dir):
        raise NotImplementedError

    @staticmethod
    def _load_csv_or_error(file_path):
        if file_path.exists():
            return FileLoader.load_csv(file_path)
        else:
            return "error"

    @staticmethod
    def _calculate_auc(submission_on_corrupted_new_data, ground_truth_csv):
        y_true = ground_truth_csv.iloc[:, -1]
        y_pred_proba = submission_on_corrupted_new_data.iloc[:, -1]
        auc = roc_auc_score(y_true, y_pred_proba)
        return auc
