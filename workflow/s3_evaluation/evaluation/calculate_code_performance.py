import argparse
import json

from tadv.utils import load_dotenv

load_dotenv()
from workflow.s3_evaluation.evaluation.metrics_calculation.calculator import MetricsCalculation

from tadv.dq_manager import DeequDataQualityManager
from tadv.utils import get_project_root


def evaluate(dataset_name, downstream_task, processed_data_label):
    dq_manager = DeequDataQualityManager()
    metric_calculator = MetricsCalculation()
    project_root = get_project_root()
    processed_data_path = project_root / "data_processed" / dataset_name / downstream_task / f"{processed_data_label}"
    output_dir = processed_data_path / "output"
    output_validation_dir = processed_data_path / "output_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_validation_dir.mkdir(parents=True, exist_ok=True)

    for script_output_dir in sorted(output_dir.iterdir()):
        print(f"evaluating script: {script_output_dir.name}")
        result = metric_calculator.calculate(downstream_task=downstream_task, script_output_dir=script_output_dir)
        output_file = output_validation_dir / f"{script_output_dir.name}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


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


    parser = argparse.ArgumentParser(description='calculate code performance')
    parser.add_argument('--dataset-option', type=str, default="all",
                        help='Dataset name. Options: 0: playground-series-s4e10, 1: healthcare_dataset')
    parser.add_argument('--downstream-task-option', type=str, default="all",
                        help='Downstream task. Options: 0: ml_inference_classification, 1: ml_inference_regression, 2: sql_query, 3: webpage_generation')
    parser.add_argument('--processed-data-label', type=str, default="0",
                        help='Version Label of the processed data')
    args = parser.parse_args()
    for dataset_name in parse_multiple_indices(args.dataset_option, dataset_name_options):
        for downstream_task in parse_multiple_indices(args.downstream_task_option, downstream_task_type_options):
            evaluate(dataset_name=dataset_name, downstream_task=downstream_task,
                     processed_data_label=args.processed_data_label)
