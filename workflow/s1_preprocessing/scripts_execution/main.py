import argparse

from workflow.s1_preprocessing.scripts_execution.run_ml_inference_code import run_ml_inference
from workflow.s1_preprocessing.scripts_execution.run_sql_code import run_sql_code
from workflow.s1_preprocessing.scripts_execution.run_web_code import run_web_code

if __name__ == "__main__":
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_type_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                                    "webpage_generation"]


    def parse_multiple_indices(input_str, options_list):
        """Parses comma-separated indices or 'all'."""
        if input_str.lower() == "all":
            return options_list
        indices = list(map(int, input_str.split(",")))
        return [options_list[i] for i in indices]


    parser = argparse.ArgumentParser(description='Run scripts for downstream tasks')
    parser.add_argument('--dataset-option', type=str, default="all",
                        help='Comma-separated dataset options or "all". Options: 0: playground-series-s4e10, 1: healthcare_dataset')
    parser.add_argument('--downstream-task-option', type=str, default="2",
                        help='Comma-separated downstream task options or "all". Options: 0: ml_inference_classification, 1: ml_inference_regression, 2: sql_query, 3: webpage_generation')
    parser.add_argument('--processed-data-label', type=str, default="0",
                        help='Version Label of the processed data')
    parser.add_argument('--single-script', type=str, default="",
                        help='Run a single script if specified, e.g., "script_name"')
    args = parser.parse_args()

    # Parse inputs
    dataset_selections = parse_multiple_indices(args.dataset_option, dataset_name_options)
    downstream_task_selections = parse_multiple_indices(args.downstream_task_option, downstream_task_type_options)

    for dataset_name_option in dataset_selections:
        for downstream_task_option in downstream_task_selections:
            processed_data_label = args.processed_data_label
            single_script = args.single_script
            if downstream_task_option in ["ml_inference_classification", "ml_inference_regression"]:
                run_ml_inference(dataset_name=dataset_name_option,
                                 downstream_task_type=downstream_task_option,
                                 processed_data_label=processed_data_label,
                                 single_script=single_script)
            elif downstream_task_option == "sql_query":
                run_sql_code(dataset_name=dataset_name_option,
                             processed_data_label=f"{processed_data_label}",
                             single_script=single_script)
            elif downstream_task_option == "webpage_generation":
                run_web_code(dataset_name=dataset_name_option,
                             processed_data_label=f"{processed_data_label}",
                             single_script=single_script)
