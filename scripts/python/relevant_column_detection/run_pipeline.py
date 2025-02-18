from cadv_exploration.utils import load_dotenv

load_dotenv()
from scripts.python.relevant_column_detection.string_matching import run_string_matching
from scripts.python.relevant_column_detection.metrics import RelevantColumnDetectionMetric
from scripts.python.utils import load_train_and_test_spark_data

from cadv_exploration.utils import get_task_instance
from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.llm.langchain.downstream_task_prompt import SQL_QUERY_TASK_DESCRIPTION, \
    ML_INFERENCE_TASK_DESCRIPTION
from cadv_exploration.utils import get_project_root


def run_langchain_cadv_on_single_model(data_name, model_name, processed_data_idx):
    dq_manager = DeequDataQualityManager()

    original_data_path = get_project_root() / "data" / f"{data_name}"

    spark_train_data, spark_train, spark_validation_data, spark_validation = load_train_and_test_spark_data(
        data_name=data_name, processed_data_idx=processed_data_idx, dq_manager=dq_manager
    )

    column_list = sorted(spark_validation_data.columns, key=lambda x: x.lower())

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_train_data, spark_train)

    metric_evaluator = RelevantColumnDetectionMetric(average='macro')
    result_each_type = {}
    for task_type in ['bi', 'dev', 'feature_engineering', 'classification', 'regression']:
        scripts_path_dir = original_data_path / "scripts" / task_group_mapping(task_type)
        print(task_type)
        all_ground_truth_vectors = []
        all_relevant_columns_vectors = []
        for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
            if task_type not in script_path.name:
                continue
            task_instance = get_task_instance(script_path)

            if model_name == "string-matching":
                relevant_columns_list = run_string_matching(column_list, task_instance.original_script)
            else:
                relevant_columns_list = run_llm(column_desc, model_name, task_instance.original_script,
                                                task_group_mapping(task_type))

            ground_truth = sorted(task_instance.annotations['required_columns'], key=lambda x: x.lower())
            ground_truth_vector, relevant_columns_vector = metric_evaluator.binary_vectorize(column_list,
                                                                                             ground_truth,
                                                                                             relevant_columns_list)
            all_ground_truth_vectors.append(ground_truth_vector)
            all_relevant_columns_vectors.append(relevant_columns_vector)
        result_each_type[task_type] = metric_evaluator.evaluate(all_ground_truth_vectors, all_relevant_columns_vectors)
    return result_each_type


def task_group_mapping(task_type):
    return {
        'bi': 'sql',
        'dev': 'sql',
        'exclude_clause': 'sql',
        'feature_engineering': 'sql',
        'classification': 'ml',
        'regression': 'ml'
    }[task_type]


def run_llm(column_desc, model_name, script_context, task_group):
    input_variables = {
        "column_desc": column_desc,
        "script": script_context,
    }
    if task_group == 'sql':
        lc = LangChainCADV(model_name=model_name, downstream_task_description=SQL_QUERY_TASK_DESCRIPTION)
    elif task_group == 'ml':
        lc = LangChainCADV(model_name=model_name, downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    else:
        raise ValueError(f"Unknown task group: {task_group}")
    max_retries = 3
    relevant_columns_list, expectations, suggestions = lc.invoke(
        input_variables=input_variables, num_stages=1, max_retries=max_retries
    )
    relevant_columns_list = sorted(relevant_columns_list, key=lambda x: x.lower())
    return relevant_columns_list


def run_langchain_cadv_on_all_models(data_name, model_names, processed_data_idx):
    result_each_model = {}
    for model_name in model_names:
        print(model_name)
        result_each_model[model_name] = run_langchain_cadv_on_single_model(data_name, model_name, processed_data_idx)
    return result_each_model


if __name__ == "__main__":
    data_name = "playground-series-s4e10"
    model_names = ["string-matching", "gpt-4o-mini", "gpt-4o", "llama3.2:1b", "llama3.2"]
    processed_data_idx = 'base_version'
    all_results = run_langchain_cadv_on_all_models(data_name, model_names, processed_data_idx)
    # reverse the order of the keys
    for task_type in ['bi', 'dev', 'feature_engineering', 'classification', 'regression']:
        result_each_model = {
            model_name: all_results[model_name][task_type]
            for model_name in model_names
        }
        RelevantColumnDetectionMetric().plot_model_metrics(
            result_each_model,
            picture_name=f"{data_name}/sql_metrics_{task_type}.png"
        )
