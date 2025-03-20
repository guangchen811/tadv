import json
from collections import defaultdict

import pandas as pd

from tadv.data_models import ValidationResults
from tadv.utils import load_dotenv, get_current_folder

load_dotenv()

from tadv.utils import get_project_root


def results_calculation(dataset_name, downstream_task, processed_data_label):
    project_root = get_project_root()
    processed_data_path = project_root / "data_processed" / dataset_name / downstream_task / f"{processed_data_label}"
    constraints_validation_dir = processed_data_path / "constraints_validation"
    output_validation_dir = processed_data_path / "output_validation"

    output_validation_dict = defaultdict(dict)
    for script_output_dir in sorted(output_validation_dir.iterdir()):
        output_file = output_validation_dir / f"{script_output_dir.name}"
        with output_file.open("r", encoding="utf-8") as f:
            result = json.load(f)
        output_validation_dict[script_output_dir.stem] = result

    output_result_data = []
    for script, results in output_validation_dict.items():
        output_result_data.append({"Script": script,
                                   "Execution Result on Clean Data": results["result_on_clean_new_data"],
                                   "Execution Result on Corrupted Data": results["result_on_corrupted_new_data"]})

    output_result_df = pd.DataFrame(output_result_data)

    constraints_validation_dict = defaultdict(dict)
    if not constraints_validation_dir.exists():
        raise FileNotFoundError(f"Constraints validation directory not found: {constraints_validation_dir}")
    deequ_result = {}
    deequ_with_column_skipped_result = {}
    for script_constraints_dir in sorted(constraints_validation_dir.iterdir()):
        if script_constraints_dir.is_file():
            # deeque constraints
            new_data_name, _ = script_constraints_dir.stem.split("__")
            constraints_file = constraints_validation_dir / f"{script_constraints_dir.name}"
            deequ_result[new_data_name] = ValidationResults.from_yaml(constraints_file).check_result()
            deequ_with_column_skipped_result[new_data_name] = ValidationResults.from_yaml(
                constraints_file).check_result(column_skipped=['person_age', 'Age'])
    for script_constraints_dir in sorted(constraints_validation_dir.iterdir()):
        if script_constraints_dir.is_dir():
            for constraints_file_name in sorted(script_constraints_dir.iterdir()):
                if constraints_file_name.suffix != ".yaml":
                    raise ValueError(f"Only yaml files are supported. Found {constraints_file_name.suffix}")
                new_data_name, method_used, strategy_used = constraints_file_name.stem.split("__")
                constraints_file = constraints_validation_dir / script_constraints_dir.name / f"{constraints_file_name}"
                constraints_validation_dict[script_constraints_dir.name][
                    f"{new_data_name}__{method_used}__{strategy_used}"] = ValidationResults.from_yaml(
                    constraints_file).check_result()
            for new_data_name in deequ_result.keys():
                constraints_validation_dict[script_constraints_dir.name][f"{new_data_name}__deequ__None"] = \
                    deequ_result[
                        new_data_name]
                constraints_validation_dict[script_constraints_dir.name][f"{new_data_name}__deequ__column_skipped"] = \
                    deequ_with_column_skipped_result[
                        new_data_name]

    constraints_result_data = []
    for script, results in constraints_validation_dict.items():
        for experiment_config, result in results.items():
            new_data_name, method_used, strategy_used = experiment_config.split("__")
            constraints_result_data.append({"Script": script, "Model": method_used, "Passed Constraints": result[0],
                                            "Failed Constraints": result[1], "Strategy": strategy_used,
                                            "New Data Type": new_data_name})
    constraints_result_df = pd.DataFrame(constraints_result_data)
    if constraints_result_df.empty:
        raise FileNotFoundError(f"No constraints validation results found in {constraints_validation_dir}")

    # Pivot the table to merge clean and corrupted rows into a single row
    constraints_result_df['New Data Type'] = constraints_result_df['New Data Type'].apply(
        lambda x: 'Clean' if 'clean' in x else 'Corrupted')
    constraints_result_df_pivot = constraints_result_df.pivot(index=['Script', 'Model', 'Strategy'],
                                                              columns='New Data Type',
                                                              values=['Passed Constraints', 'Failed Constraints'])

    # Flatten MultiIndex columns
    constraints_result_df_pivot.columns = [f"{col[0]} ({col[1]})" for col in constraints_result_df_pivot.columns]
    constraints_result_df_pivot.reset_index(inplace=True)

    joined_df = pd.merge(output_result_df, constraints_result_df_pivot, how="outer", on="Script")
    result_table_path = get_current_folder().parent / "result_tables/"
    result_table_path.mkdir(parents=True, exist_ok=True)
    joined_df.to_csv(result_table_path / f"{dataset_name}__{downstream_task}__{processed_data_label}.csv")


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


    parser = argparse.ArgumentParser(description='Results Calculation')
    parser.add_argument('--dataset-option', type=str, default="all",
                        help='Dataset name. Options: 0: playground-series-s4e10, 1: healthcare_dataset')
    parser.add_argument('--downstream-task-option', type=str, default="all",
                        help='Downstream task. Options: 0: ml_inference_classification, 1: ml_inference_regression, 2: sql_query, 3: webpage_generation')
    parser.add_argument('--processed-data-label', type=str, default="0",
                        help='Version Label of the processed data')
    args = parser.parse_args()

    dataset_selections = parse_multiple_indices(args.dataset_option, dataset_name_options)
    downstream_task_selections = parse_multiple_indices(args.downstream_task_option, downstream_task_type_options)

    for dataset_name in dataset_selections:
        for downstream_task in downstream_task_selections:
            try:
                results_calculation(dataset_name, downstream_task, args.processed_data_label)
            except FileNotFoundError as e:
                print(f"Skipping {dataset_name}__{downstream_task} due to error: {e}")
                continue
            print(f"Results calculated for {dataset_name}__{downstream_task}")
