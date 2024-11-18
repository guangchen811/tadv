import pandas as pd

from cadv_exploration.utils import load_dotenv
from data_models import Constraints

load_dotenv()

from sklearn.metrics import roc_auc_score
import numpy as np
from cadv_exploration.deequ_wrapper import DeequWrapper
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root
import oyaml as yaml


def evaluate_playground_series_s4e10(processed_data_idx):
    deequ_wrapper = DeequWrapper()
    project_root = get_project_root()
    processed_data_path = project_root / "data_processed" / "playground-series-s4e10" / f"{processed_data_idx}"
    ground_truth = FileLoader.load_csv(processed_data_path / "files_with_clean_test_data" / "ground_truth.csv")
    scripts_output_dir = processed_data_path / "constraints"
    clean_test_data = FileLoader.load_csv(processed_data_path / "files_with_clean_test_data" / "test.csv")
    corrupted_test_data = FileLoader.load_csv(processed_data_path / "files_with_corrupted_test_data" / "test.csv")
    for script_output_dir in scripts_output_dir.iterdir():
        print(f"evaluating script: {script_output_dir.name}")

        cadv_suggestion_file_path = script_output_dir / "cadv_constraints.yaml"
        cadv_constraints = Constraints.from_yaml(cadv_suggestion_file_path)
        valid_code_column_map = cadv_constraints.get_suggestions_code_column_map(valid_only=True)
        code_list_for_cadv_constraints = [item for item in valid_code_column_map.keys()]
        # Validate the constraints on the before broken data
        spark_clean_test_data, spark_clean_test = deequ_wrapper.spark_df_from_pandas_df(clean_test_data)
        status_on_clean_text_data = deequ_wrapper.validate_on_df(spark_clean_test, spark_clean_test_data,
                                                                 code_list_for_cadv_constraints)
        # Validate the constraints on the after broken data
        spark_corrupted_test_data, spark_corrupted_test = deequ_wrapper.spark_df_from_pandas_df(corrupted_test_data)
        status_on_corrupted_text_data = deequ_wrapper.validate_on_df(spark_corrupted_test, spark_corrupted_test_data,
                                                                     code_list_for_cadv_constraints)

        spark_clean_test.sparkContext._gateway.shutdown_callback_server()
        spark_corrupted_test.sparkContext._gateway.shutdown_callback_server()
        spark_clean_test.stop()
        spark_corrupted_test.stop()

        deequ_suggestion_file_path = script_output_dir.parent / "deequ_constraints.yaml"
        deequ_constraints = Constraints.from_yaml(deequ_suggestion_file_path)
        deequ_valid_code_column_map = deequ_constraints.get_suggestions_code_column_map(valid_only=True)
        code_list_for_deequ_constraints = [item for item in deequ_valid_code_column_map.keys()]
        # Validate the constraints on the before broken data
        spark_clean_test_data, spark_clean_test = deequ_wrapper.spark_df_from_pandas_df(clean_test_data)
        status_on_clean_text_data_deequ = deequ_wrapper.validate_on_df(spark_clean_test, spark_clean_test_data,
                                                                       code_list_for_deequ_constraints)
        # Validate the constraints on the after broken data
        spark_corrupted_test_data, spark_corrupted_test = deequ_wrapper.spark_df_from_pandas_df(corrupted_test_data)
        status_on_corrupted_text_data_deequ = deequ_wrapper.validate_on_df(spark_corrupted_test,
                                                                           spark_corrupted_test_data,
                                                                           code_list_for_deequ_constraints)

        spark_clean_test.sparkContext._gateway.shutdown_callback_server()
        spark_corrupted_test.sparkContext._gateway.shutdown_callback_server()
        spark_clean_test.stop()
        spark_corrupted_test.stop()

        result_on_clean_test_data = FileLoader.load_csv(
            script_output_dir.parent.parent / "output" / script_output_dir.name / "results_on_clean_test_data" / "submission.csv")
        auc_on_clean_test_data = roc_auc_score(np.where(ground_truth['loan_status'].to_numpy() > 0.5, 1, 0),
                                               result_on_clean_test_data['loan_status'].to_numpy())
        try:
            result_on_corrupted_test_data = FileLoader.load_csv(
                script_output_dir.parent.parent / "output" / script_output_dir.name / "results_on_corrupted_test_data" / "submission.csv")
            auc_on_corrupted_test_data = roc_auc_score(np.where(ground_truth['loan_status'].to_numpy() > 0.5, 1, 0),
                                                       result_on_corrupted_test_data['loan_status'].to_numpy())
        except Exception as e:
            auc_on_corrupted_test_data = np.nan
        print(
            f"AUC: {auc_on_clean_test_data}")
        print(
            f"AUC: {auc_on_corrupted_test_data}")

        print(status_on_clean_text_data.count("Success") / len(status_on_clean_text_data))
        print(status_on_corrupted_text_data.count("Success") / len(status_on_corrupted_text_data))

        print(status_on_clean_text_data_deequ.count("Success") / len(status_on_clean_text_data_deequ))
        print(status_on_corrupted_text_data_deequ.count("Success") / len(status_on_corrupted_text_data_deequ))

        evaluate_file_path = script_output_dir.parent.parent / "output" / script_output_dir.name / "evaluation.csv"
        pd.DataFrame({
            "auc_on_clean_test_data": [auc_on_clean_test_data],
            "auc_on_corrupted_test_data": [auc_on_corrupted_test_data],
            "status_on_clean_text_data": [status_on_clean_text_data.count("Success") / len(status_on_clean_text_data)],
            "status_on_corrupted_text_data": [
                status_on_corrupted_text_data.count("Success") / len(status_on_corrupted_text_data)],
            "status_on_clean_text_data_deequ": [
                status_on_clean_text_data_deequ.count("Success") / len(status_on_clean_text_data_deequ)],
            "status_on_corrupted_text_data_deequ": [
                status_on_corrupted_text_data_deequ.count("Success") / len(status_on_corrupted_text_data_deequ)]
        }).to_csv(evaluate_file_path, index=False)


if __name__ == "__main__":
    evaluate_playground_series_s4e10(processed_data_idx=0)
