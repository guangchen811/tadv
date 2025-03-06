from cadv_exploration.utils import load_dotenv

load_dotenv()
from inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from scripts.python.relevant_column_detection.run_pipeline import run_llm
from scripts.python.utils import load_train_and_test_spark_data, load_previous_and_new_spark_data
from utils import get_project_root, get_task_instance

from cadv_exploration.dq_manager import DeequDataQualityManager


def init_relevant_column(processed_idx, dataset_name, downstream_task_type, single_script=""):
    dq_manager = DeequDataQualityManager()
    project_root = get_project_root()
    downstream_task_type_script_dir_name_mapping = {
        "ml_inference_classification": "ml_inference",
        "ml_inference_regression": "ml_inference",
        "dev": "sql_query",
        "bi": "sql_query",
        "feature_engineering": "sql_query",
        "info": "webpage_generation"
    }
    downstream_task_type_path_mapping = {
        "ml_inference_classification": "ml_inference_classification",
        "ml_inference_regression": "ml_inference_regression",
        "dev": "sql_query",
        "bi": "sql_query",
        "feature_engineering": "sql_query",
        "web": "web",
        "info": "web"
    }
    original_data_path = project_root / "data" / f"{dataset_name}"
    processed_data_path = project_root / "data_processed" / f"{dataset_name}_{downstream_task_type_path_mapping[downstream_task_type]}" / processed_idx
    script_dir = original_data_path / "scripts" / downstream_task_type_script_dir_name_mapping[downstream_task_type]

    if downstream_task_type_script_dir_name_mapping[downstream_task_type] == "ml":
        spark_previous_data, spark_previous, spark_new_data, spark_new = load_train_and_test_spark_data(
            processed_data_name=f"{dataset_name}_{downstream_task_type_path_mapping[downstream_task_type]}",
            processed_data_label=processed_idx, dq_manager=dq_manager
        )
    else:
        spark_previous_data, spark_previous, spark_new_data, spark_new = load_previous_and_new_spark_data(
            processed_data_name=f"{dataset_name}_{downstream_task_type_path_mapping[downstream_task_type]}",
            processed_data_label=processed_idx, dq_manager=dq_manager)

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_previous_data, spark_previous)
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        if downstream_task_type_script_dir_name_mapping[downstream_task_type] == "ml" and script_path.name.split("_")[
            0] != downstream_task_type.rsplit("_", 1)[-1]:
            continue

        task_instance = get_task_instance(script_path)
        relevant_columns_list = run_llm(column_desc, "gpt-4o", task_instance.original_script,
                                        downstream_task_type_script_dir_name_mapping[downstream_task_type])
        relevant_columns_result_path = processed_data_path / "relevant_columns" / script_path.stem / "relevant_columns.txt"
        relevant_columns_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(relevant_columns_result_path, "w") as f:
            for col in relevant_columns_list:
                f.write(f"{col}\n")


if __name__ == "__main__":
    # dataset_name = "playground-series-s4e10"
    dataset_name = "healthcare_dataset"
    downstream_task_type = "ml_inference_regression"
    # downstream_task_type = "dev"
    # downstream_task_type = "ml_inference_classification"
    processed_idx = "base_version"
    init_relevant_column(dataset_name=dataset_name, downstream_task_type=downstream_task_type,
                         processed_idx=processed_idx)
