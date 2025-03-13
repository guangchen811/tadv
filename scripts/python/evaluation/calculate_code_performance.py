import json

from tadv.utils import load_dotenv

load_dotenv()
from scripts.python.evaluation.metrics_calculation.calculator import MetricsCalculation

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
        result = metric_calculator.calculate(dataset_name, downstream_task, processed_data_label, script_output_dir)
        output_file = output_validation_dir / f"{script_output_dir.name}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_type_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                                    "webpage_generation"]

    evaluate(dataset_name=dataset_name_options[0],
             downstream_task=downstream_task_type_options[0],
             processed_data_label="0")
