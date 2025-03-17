import json
from collections import defaultdict

from tadv.data_models import ValidationResults
from tadv.utils import load_dotenv

load_dotenv()

from tadv.dq_manager import DeequDataQualityManager
from tadv.utils import get_project_root


def results_calculation(dataset_name, downstream_task, processed_data_label):
    dq_manager = DeequDataQualityManager()
    project_root = get_project_root()
    processed_data_path = project_root / "data_processed" / dataset_name / downstream_task / f"{processed_data_label}"
    constraints_validation_dir = processed_data_path / "constraints_validation"
    output_validation_dir = processed_data_path / "output_validation"

    output_validation_dict = defaultdict(dict)
    for script_output_dir in sorted(output_validation_dir.iterdir()):
        print(f"evaluating script: {script_output_dir.name}")
        output_file = output_validation_dir / f"{script_output_dir.name}"
        with output_file.open("r", encoding="utf-8") as f:
            result = json.load(f)
        output_validation_dict[script_output_dir.stem] = result

    constraints_validation_dict = defaultdict(dict)
    for script_constraints_dir in sorted(constraints_validation_dir.iterdir()):
        if script_constraints_dir.is_file():
            # read yaml
            new_data_name, _ = script_constraints_dir.stem.split("__")
            constraints_file = constraints_validation_dir / f"{script_constraints_dir.name}"
            constraints_validation_dict["deequ"][f"{new_data_name}__deequ"] = ValidationResults.from_yaml(
                constraints_file).check_result()


        else:
            for constraints_file_name in sorted(script_constraints_dir.iterdir()):
                if constraints_file_name.suffix != ".yaml":
                    raise ValueError(f"Only yaml files are supported. Found {constraints_file_name.suffix}")
                new_data_name, llm_used, strategy_used = constraints_file_name.stem.split("__")
                constraints_file = constraints_validation_dir / script_constraints_dir.name / f"{constraints_file_name}"
                constraints_validation_dict[script_constraints_dir.name][
                    f"{new_data_name}__{llm_used}__{strategy_used}"] = ValidationResults.from_yaml(
                    constraints_file).check_result()
    print(output_validation_dict)
    print(constraints_validation_dict)


if __name__ == "__main__":
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_type_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                                    "webpage_generation"]

    dataset_option = 0
    downstream_task_option = 0
    processed_data_label = "0"

    results_calculation(dataset_name=dataset_name_options[dataset_option],
                        downstream_task=downstream_task_type_options[downstream_task_option],
                        processed_data_label=processed_data_label)
