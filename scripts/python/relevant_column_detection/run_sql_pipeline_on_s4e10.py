from cadv_exploration.utils import load_dotenv

load_dotenv()
from scripts.python.relevant_column_detection.metrics import RelevantColumnDetectionMetric
from scripts.python.utils import load_train_and_test_spark_data

import importlib.util

from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from llm.langchain import LangChainCADV
from cadv_exploration.llm.langchain.downstream_task_prompt import SQL_QUERY_TASK_DESCRIPTION
from utils import get_project_root


def run_langchain_cadv_on_single_model(model_name, data_name, processed_data_idx):
    dq_manager = DeequDataQualityManager()

    original_data_path = get_project_root() / "data" / f"{data_name}"
    processed_data_path = get_project_root() / "data_processed" / f"{data_name}" / f"{processed_data_idx}"

    spark_train_data, spark_train, spark_validation_data, spark_validation = load_train_and_test_spark_data(
        data_name=data_name, processed_data_idx=processed_data_idx, dq_manager=dq_manager
    )

    column_list = sorted(spark_validation_data.columns, key=lambda x: x.lower())

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_train_data, spark_train)

    scripts_path_dir = original_data_path / "scripts_sql_labeled"

    metric_evaluator = RelevantColumnDetectionMetric(average='macro')
    result_each_type = {}
    for sql_type in ['bi', 'dev', 'exclude_clause', 'feature_engineering']:
        print(sql_type)
        all_ground_truth_vectors = []
        all_relevant_columns_vectors = []
        for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
            module_name = script_path.stem
            if not script_path.name.endswith(".py") or not module_name.startswith(sql_type):
                continue
            result_path = processed_data_path / "constraints" / f"{script_path.name.split('.')[0]}" / "cadv_constraints.yaml"
            result_path.parent.mkdir(parents=True, exist_ok=True)

            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            task_class = getattr(module, "KaggleLoanColumnDetectionTask")
            task_instance = task_class()
            script_context = task_instance.original_code
            input_variables = {
                "column_desc": column_desc,
                "script": script_context,
            }

            lc = LangChainCADV(model_name=model_name, downstream_task_description=SQL_QUERY_TASK_DESCRIPTION)

            max_retries = 3
            relevant_columns_list, expectations, suggestions = lc.invoke(
                input_variables=input_variables, num_stages=1, max_retries=max_retries
            )
            print(module_name)
            ground_truth = sorted(task_instance.required_columns(), key=lambda x: x.lower())
            relevant_columns_list = sorted(relevant_columns_list, key=lambda x: x.lower())

            ground_truth_vector, relevant_columns_vector = metric_evaluator.binary_vectorize(column_list,
                                                                                             ground_truth,
                                                                                             relevant_columns_list)
            all_ground_truth_vectors.append(ground_truth_vector)
            all_relevant_columns_vectors.append(relevant_columns_vector)
        result_each_type[sql_type] = metric_evaluator.evaluate(all_ground_truth_vectors, all_relevant_columns_vectors)
    return result_each_type


def run_langchain_cadv_on_all_models(model_names, data_name, processed_data_idx):
    result_each_model = {}
    for model_name in model_names:
        print(model_name)
        result_each_model[model_name] = run_langchain_cadv_on_single_model(model_name, data_name, processed_data_idx)
    return result_each_model


if __name__ == "__main__":
    model_names = ["llama3.2:1b", "llama3.2", "gpt-4o-mini", "gpt-4o"]
    data_name = "playground-series-s4e10"
    processed_data_idx = 8
    all_results = run_langchain_cadv_on_all_models(model_names, data_name, processed_data_idx)
    # reverse the order of the keys
    for sql_type in ['bi', 'dev', 'exclude_clause', 'feature_engineering']:
        result_each_model = {
            model_name: all_results[model_name][sql_type]
            for model_name in model_names
        }
        RelevantColumnDetectionMetric().plot_model_metrics(
            result_each_model,
            picture_name=f"sql_metrics_{sql_type}.png"
        )
