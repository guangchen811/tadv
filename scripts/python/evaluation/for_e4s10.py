import pandas as pd
from pandas.core.computation.expressions import evaluate

from cadv_exploration.utils import load_dotenv

load_dotenv()

from sklearn.metrics import roc_auc_score
import numpy as np
from cadv_exploration.deequ import spark_df_from_pandas_df
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root
from deequ._constraint_validation import validate_on_df
import oyaml as yaml


def evaluate_playground_series_s4e10():
    project_root = get_project_root()
    local_data_path = project_root / "data" / "playground-series-s4e10"
    ground_truth = FileLoader.load_csv(local_data_path / "files_with_clean_test_data" / "ground_truth.csv")
    scripts_output_dir = local_data_path / "output"
    clean_test_data = FileLoader.load_csv(local_data_path / "files_with_clean_test_data" / "test.csv")
    corrupted_test_data = FileLoader.load_csv(local_data_path / "files_with_corrupted_test_data" / "test.csv")
    for script_output_dir in scripts_output_dir.iterdir():
        print(f"evaluating script: {script_output_dir.name}")

        cadv_suggestion_file_path = script_output_dir / "cadv_constraints.yaml"
        with open(cadv_suggestion_file_path, "r") as file:
            cadv_suggestions = yaml.safe_load(file)
        code_list_for_constraints = [item for k, v in cadv_suggestions['constraints'].items() for item in v['code']]
        code_list_for_constraints = [item[0] for item in code_list_for_constraints if item[1] == "Valid"]
        # Validate the constraints on the before broken data
        spark_clean_test_data, spark_clean_test = spark_df_from_pandas_df(clean_test_data)
        status_on_clean_text_data = validate_on_df(code_list_for_constraints, spark_clean_test,
                                                   spark_clean_test_data)
        # Validate the constraints on the after broken data
        spark_corrupted_test_data, spark_corrupted_test = spark_df_from_pandas_df(corrupted_test_data)
        status_on_corrupted_text_data = validate_on_df(code_list_for_constraints, spark_corrupted_test,
                                                       spark_corrupted_test_data)

        spark_clean_test.sparkContext._gateway.shutdown_callback_server()
        spark_corrupted_test.sparkContext._gateway.shutdown_callback_server()
        spark_clean_test.stop()
        spark_corrupted_test.stop()

        deequ_suggestion_file_path = script_output_dir.parent / "deequ_constraints.yaml"
        with open(deequ_suggestion_file_path, "r") as file:
            deequ_suggestions = yaml.safe_load(file)
        code_list_for_constraints = [item[0] for item in deequ_suggestions['constraints'] if item[1] == "Valid"]

        # Validate the constraints on the before broken data
        spark_clean_test_data, spark_clean_test = spark_df_from_pandas_df(clean_test_data)
        status_on_clean_text_data_deequ = validate_on_df(code_list_for_constraints, spark_clean_test,
                                                         spark_clean_test_data)
        # Validate the constraints on the after broken data
        spark_corrupted_test_data, spark_corrupted_test = spark_df_from_pandas_df(corrupted_test_data)
        status_on_corrupted_text_data_deequ = validate_on_df(code_list_for_constraints, spark_corrupted_test,
                                                             spark_corrupted_test_data)

        spark_clean_test.sparkContext._gateway.shutdown_callback_server()
        spark_corrupted_test.sparkContext._gateway.shutdown_callback_server()
        spark_clean_test.stop()
        spark_corrupted_test.stop()
        result_on_clean_test_data = FileLoader.load_csv(script_output_dir / "results_on_clean_test_data" / "submission.csv")
        auc_on_clean_test_data = roc_auc_score(np.where(ground_truth['loan_status'].to_numpy() > 0.5, 1, 0),
                                               result_on_clean_test_data['loan_status'].to_numpy())
        try:
            result_on_corrupted_test_data = FileLoader.load_csv(
                script_output_dir / "results_on_corrupted_test_data" / "submission.csv")
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

        evaluate_file_path = script_output_dir / "evaluation.csv"
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
    evaluate_playground_series_s4e10()
