from tadv.utils import load_dotenv, get_current_folder

load_dotenv()
from scripts.python.relevant_column_detection.string_matching import run_string_matching_for_rcd, run_llm_for_rcd
from scripts.python.relevant_column_detection.metrics import RelevantColumnDetectionMetric
from scripts.python.utils import load_previous_and_new_spark_data

from tadv.utils import get_task_instance
from tadv.dq_manager import DeequDataQualityManager
from tadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from tadv.utils import get_project_root

task_group_mapping = {
    'bi': 'sql_query',
    'dev': 'sql_query',
    'feature_engineering': 'sql_query',
    'classification': 'ml_inference',
    'regression': 'ml_inference',
    'info': 'webpage_generation'
}


def run_langchain_cadv_on_single_model(dataset_name, model_name, processed_data_label):
    dq_manager = DeequDataQualityManager()

    original_data_path = get_project_root() / "data" / f"{dataset_name}"

    spark_previous_data, spark_previous, _, _ = load_previous_and_new_spark_data(
        dataset_name=dataset_name,
        downstream_task="sql_query", # it doesn't matter what this is set to as long as it's a valid task
        processed_data_label=processed_data_label,
        dq_manager=dq_manager
    )

    column_list = sorted(spark_previous_data.columns, key=lambda x: x.lower())

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_previous_data, spark_previous)

    metric_evaluator = RelevantColumnDetectionMetric(average='macro')
    result_each_type = {}

    result_path = get_current_folder() / "relevant_columns" / f"{dataset_name}" / f"{model_name}"
    result_path.mkdir(parents=True, exist_ok=True)
    for task_type in ['bi', 'dev', 'feature_engineering', 'classification', 'regression', 'info']:
        scripts_path_dir = original_data_path / "scripts" / task_group_mapping[task_type]
        print(task_type, end=' ')
        all_ground_truth_vectors = []
        all_relevant_columns_vectors = []
        for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
            if task_type not in script_path.name and task_type != 'info':
                continue
            task_instance = get_task_instance(script_path)

            if model_name == "string-matching":
                relevant_columns_list = run_string_matching_for_rcd(column_list, task_instance.original_script)
            else:
                relevant_columns_list = run_llm_for_rcd(column_desc, model_name, task_instance.original_script,
                                                        task_group_mapping[task_type])

            save_dir = result_path / f"relevant_columns__{script_path.stem}.txt"
            with open(save_dir, 'a') as f:
                for column in relevant_columns_list:
                    f.write(f"{column}\n")

            ground_truth = sorted(task_instance.annotations['required_columns'], key=lambda x: x.lower())
            ground_truth_vector, relevant_columns_vector = metric_evaluator.binary_vectorize(column_list,
                                                                                             ground_truth,
                                                                                             relevant_columns_list)
            all_ground_truth_vectors.append(ground_truth_vector)
            all_relevant_columns_vectors.append(relevant_columns_vector)
        result_each_type[task_type] = [all_ground_truth_vectors, all_relevant_columns_vectors]
    print("done")
    return result_each_type


def run_langchain_cadv_on_all_models(dataset_name, model_names, processed_data_label):
    result_each_model = {}
    for model_name in model_names:
        print(model_name, end=': ')
        result_each_model[model_name] = run_langchain_cadv_on_single_model(dataset_name, model_name,
                                                                           processed_data_label)
    return result_each_model


if __name__ == "__main__":
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    # model_names = ["string-matching", "llama3.2:1b", "llama3.2", "gpt-4o-mini", "gpt-4o"]
    model_names = ["string-matching", "gpt-3.5-turbo", "gpt-4o", "gpt-4.5-preview"]
    processed_data_label = '0'
    dataset_name = dataset_name_options[0]

    all_results = run_langchain_cadv_on_all_models(dataset_name=dataset_name,
                                                   model_names=model_names,
                                                   processed_data_label=processed_data_label)
