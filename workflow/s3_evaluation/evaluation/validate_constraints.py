from tadv.data_models import Constraints, ValidationResults
from tadv.utils import load_dotenv

load_dotenv()

from tadv.dq_manager import DeequDataQualityManager
from tadv.loader import FileLoader
from tadv.utils import get_project_root


def evaluate(dataset_name, downstream_task, processed_data_label):
    dq_manager = DeequDataQualityManager()
    project_root = get_project_root()
    processed_data_path = project_root / "data_processed" / dataset_name / downstream_task / f"{processed_data_label}"
    constraints_validation_dir = processed_data_path / "constraints_validation"
    constraints_validation_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir = processed_data_path / "constraints"

    if downstream_task in ["ml_inference_classification", "ml_inference_regression"]:
        clean_test_data = FileLoader.load_csv(processed_data_path / "files_with_clean_new_data" / "test.csv")
        corrupted_test_data = FileLoader.load_csv(processed_data_path / "files_with_corrupted_new_data" / "test.csv")
    elif downstream_task in ["sql_query", "webpage_generation"]:
        clean_test_data = FileLoader.load_csv(processed_data_path / "files_with_clean_new_data" / "new_data.csv")
        corrupted_test_data = FileLoader.load_csv(
            processed_data_path / "files_with_corrupted_new_data" / "new_data.csv")
    else:
        raise ValueError(f"Downstream task {downstream_task} is not supported")

    deequ_suggestion_file_path = constraints_dir / "deequ_constraints.yaml"
    validation_results_on_clean_test_data_deequ, validation_results_on_corrupted_test_data_deequ = validate_on_both_test_data(
        deequ_suggestion_file_path,
        clean_test_data,
        corrupted_test_data,
        dq_manager)

    validation_results_on_clean_test_data_deequ.save_to_yaml(
        constraints_validation_dir / "validation_results_on_clean_test_data__deequ.yaml")
    validation_results_on_corrupted_test_data_deequ.save_to_yaml(
        constraints_validation_dir / "validation_results_on_corrupted_test_data__deequ.yaml")

    for script_constraints_dir in constraints_dir.iterdir():
        if script_constraints_dir.is_file():
            continue
        print(f"evaluating script: {script_constraints_dir.name}")
        for constraints_file_name in script_constraints_dir.iterdir():
            if constraints_file_name.suffix != ".yaml":
                raise ValueError(f"Only yaml files are supported. Found {constraints_file_name.suffix}")
            tadv_suggestion_file_path = script_constraints_dir / constraints_file_name
            _, llm_used, strategy_used = constraints_file_name.stem.split("__")
            validation_results_on_clean_test_data_tadv, validation_results_on_corrupted_test_data_tadv = validate_on_both_test_data(
                tadv_suggestion_file_path,
                clean_test_data,
                corrupted_test_data,
                dq_manager)
            (constraints_validation_dir / script_constraints_dir.name).mkdir(parents=True, exist_ok=True)
            validation_results_on_clean_test_data_tadv.save_to_yaml(
                constraints_validation_dir / script_constraints_dir.name / f"validation_results_on_clean_test_data__{llm_used}__{strategy_used}.yaml")
            validation_results_on_corrupted_test_data_tadv.save_to_yaml(
                constraints_validation_dir / script_constraints_dir.name / f"validation_results_on_corrupted_test_data__{llm_used}__{strategy_used}.yaml")


def validate_on_both_test_data(suggestion_file_path, clean_test_data, corrupted_test_data, dq_manager):
    constraints = Constraints.from_yaml(suggestion_file_path)
    valid_code_column_map = constraints.get_suggestions_code_column_map(valid_only=True)
    code_list_for_constraints = [item for item in valid_code_column_map.keys()]
    # Validate the constraints on the before broken data
    spark_clean_test_data, spark_clean_test = dq_manager.spark_df_from_pandas_df(clean_test_data)
    status_on_clean_test_data = dq_manager.validate_on_spark_df(spark_clean_test, spark_clean_test_data,
                                                                code_list_for_constraints)
    validation_results_dict_on_clean_test_data = build_validation_results_dict(code_list_for_constraints,
                                                                               status_on_clean_test_data,
                                                                               valid_code_column_map)
    validation_results_on_clean_test_data = ValidationResults.from_dict(validation_results_dict_on_clean_test_data)
    # Validate the constraints on the after broken data
    spark_corrupted_test_data, spark_corrupted_test = dq_manager.spark_df_from_pandas_df(corrupted_test_data)
    status_on_corrupted_test_data = dq_manager.validate_on_spark_df(spark_corrupted_test,
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
    import argparse

    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_type_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                                    "webpage_generation"]


    def parse_multiple_indices(input_str, options_list):
        """Parses comma-separated indices or 'all'."""
        if input_str.lower() == "all":
            return options_list
        indices = list(map(int, input_str.split(",")))
        return [options_list[i] for i in indices]


    parser = argparse.ArgumentParser(description='validate constraints')
    parser.add_argument('--dataset-option', type=str, default="all",
                        help='Dataset name. Options: 0: playground-series-s4e10, 1: healthcare_dataset')
    parser.add_argument('--downstream-task-option', type=str, default="all",
                        help='Downstream task. Options: 0: ml_inference_classification, 1: ml_inference_regression, 2: sql_query, 3: webpage_generation')
    parser.add_argument('--processed-data-label', type=str, default="0",
                        help='Version Label of the processed data')
    args = parser.parse_args()

    for dataset_option in parse_multiple_indices(args.dataset_option, dataset_name_options):
        for downstream_task_option in parse_multiple_indices(args.downstream_task_option, downstream_task_type_options):
            evaluate(dataset_name=dataset_option, downstream_task=downstream_task_option,
                     processed_data_label=args.processed_data_label)
