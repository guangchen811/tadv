import pandas as pd

from cadv_exploration.utils import load_dotenv
from data_models import Constraints, ValidationResults

load_dotenv()

from sklearn.metrics import roc_auc_score
import numpy as np
from cadv_exploration.deequ_wrapper import DeequWrapper
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root


def evaluate_playground_series_s4e10(processed_data_idx):
    deequ_wrapper = DeequWrapper()
    project_root = get_project_root()
    processed_data_path = project_root / "data_processed" / "playground-series-s4e10" / f"{processed_data_idx}"
    ground_truth = FileLoader.load_csv(processed_data_path / "files_with_clean_test_data" / "ground_truth.csv")
    output_dir = processed_data_path / "output"
    constraints_dir = processed_data_path / "constraints"
    clean_test_data = FileLoader.load_csv(processed_data_path / "files_with_clean_test_data" / "test.csv")
    corrupted_test_data = FileLoader.load_csv(processed_data_path / "files_with_corrupted_test_data" / "test.csv")

    deequ_suggestion_file_path = constraints_dir / "deequ_constraints.yaml"
    validation_results_on_clean_test_data_deequ, validation_results_on_corrupted_test_data_deequ = validate_on_both_test_data(
        deequ_suggestion_file_path,
        clean_test_data,
        corrupted_test_data,
        deequ_wrapper)

    validation_results_on_clean_test_data_deequ.save_to_yaml(
        output_dir / "validation_results_on_clean_test_data_deequ.yaml")
    validation_results_on_corrupted_test_data_deequ.save_to_yaml(
        output_dir / "validation_results_on_corrupted_test_data_deequ.yaml")

    for script_constraints_dir in sorted(constraints_dir.iterdir(), key=lambda x: x.name, reverse=True):
        if script_constraints_dir.is_file():
            continue
        print(f"evaluating script: {script_constraints_dir.name}")
        cadv_suggestion_file_path = script_constraints_dir / "cadv_constraints.yaml"
        validation_results_on_clean_test_data_cadv, validation_results_on_corrupted_test_data_cadv = validate_on_both_test_data(
            cadv_suggestion_file_path,
            clean_test_data,
            corrupted_test_data,
            deequ_wrapper)

        validation_results_on_clean_test_data_cadv.save_to_yaml(
            output_dir / script_constraints_dir.name / "validation_results_on_clean_test_data_cadv.yaml")
        validation_results_on_corrupted_test_data_cadv.save_to_yaml(
            output_dir / script_constraints_dir.name / "validation_results_on_corrupted_test_data_cadv.yaml")

        # result_on_clean_test_data = FileLoader.load_csv(
        #     script_constraints_dir.parent.parent / "output" / script_constraints_dir.name / "results_on_clean_test_data" / "submission.csv")
        # auc_on_clean_test_data = roc_auc_score(np.where(ground_truth['loan_status'].to_numpy() > 0.5, 1, 0),
        #                                        result_on_clean_test_data.iloc[:, 1].to_numpy())
        # try:
        #     result_on_corrupted_test_data = FileLoader.load_csv(
        #         script_constraints_dir.parent.parent / "output" / script_constraints_dir.name / "results_on_corrupted_test_data" / "submission.csv")
        #     auc_on_corrupted_test_data = roc_auc_score(np.where(ground_truth['loan_status'].to_numpy() > 0.5, 1, 0),
        #                                                result_on_corrupted_test_data.iloc[:, 1].to_numpy())
        # except Exception as e:
        #     auc_on_corrupted_test_data = np.nan
        # model_performance = pd.DataFrame({
        #     "auc_on_clean_test_data": [auc_on_clean_test_data],
        #     "auc_on_corrupted_test_data": [auc_on_corrupted_test_data],
        # })
        # model_performance.to_csv(output_dir / script_constraints_dir.name / "model_performance.csv", index=False)


def validate_on_both_test_data(suggestion_file_path, clean_test_data, corrupted_test_data, deequ_wrapper):
    constraints = Constraints.from_yaml(suggestion_file_path)
    valid_code_column_map = constraints.get_suggestions_code_column_map(valid_only=True)
    code_list_for_constraints = [item for item in valid_code_column_map.keys()]
    # Validate the constraints on the before broken data
    spark_clean_test_data, spark_clean_test = deequ_wrapper.spark_df_from_pandas_df(clean_test_data)
    status_on_clean_test_data = deequ_wrapper.validate_on_df(spark_clean_test, spark_clean_test_data,
                                                             code_list_for_constraints)
    validation_results_dict_on_clean_test_data = build_validation_results_dict(code_list_for_constraints,
                                                                               status_on_clean_test_data,
                                                                               valid_code_column_map)
    validation_results_on_clean_test_data = ValidationResults.from_dict(validation_results_dict_on_clean_test_data)
    # Validate the constraints on the after broken data
    spark_corrupted_test_data, spark_corrupted_test = deequ_wrapper.spark_df_from_pandas_df(corrupted_test_data)
    status_on_corrupted_test_data = deequ_wrapper.validate_on_df(spark_corrupted_test,
                                                                 spark_corrupted_test_data,
                                                                 code_list_for_constraints)
    validation_results_dict_on_corrupted_test_data = build_validation_results_dict(code_list_for_constraints,
                                                                                   status_on_corrupted_test_data,
                                                                                   valid_code_column_map)
    validation_results_on_corrupted_test_data = ValidationResults.from_dict(
        validation_results_dict_on_corrupted_test_data)
    spark_clean_test.sparkContext._gateway.shutdown_callback_server()
    spark_corrupted_test.sparkContext._gateway.shutdown_callback_server()
    spark_clean_test.stop()
    spark_corrupted_test.stop()
    return validation_results_on_clean_test_data, validation_results_on_corrupted_test_data


def build_validation_results_dict(code_list_for_constraints, status_on_clean_test_data, valid_code_column_map):
    code_status_map = {code_list_for_constraints[i]: status_on_clean_test_data[i] for i in
                       range(len(code_list_for_constraints))}
    validation_results_dict = {"results": {column: {"code": []} for column in valid_code_column_map.values()}}
    for code, column in valid_code_column_map.items():
        validation_results_dict["results"][column]["code"].append(
            [code, "Passed" if code_status_map[code] == "Success" else "Failed"])
    return validation_results_dict


if __name__ == "__main__":
    evaluate_playground_series_s4e10(processed_data_idx=8)
