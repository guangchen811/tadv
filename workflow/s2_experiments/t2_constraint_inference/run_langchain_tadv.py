from tadv.utils import load_dotenv

load_dotenv()

from tadv.utils import get_task_instance
from tadv.utils import get_project_root
from tadv.data_models import Constraints
from tadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from tadv.dq_manager import DeequDataQualityManager
from tadv.llm.langchain import LangChainCADV
from tadv.llm.langchain.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION, \
    SQL_QUERY_TASK_DESCRIPTION, WEB_TASK_DESCRIPTION
from workflow.s2_experiments.utils import setup_logger, load_train_and_test_spark_data, load_previous_and_new_spark_data


def run_langchain_cadv(dataset_name, downstream_task, model_name, processed_data_label, assumption_generation_trick,
                       single_script_name=None):
    dq_manager = DeequDataQualityManager()
    logger = setup_logger(get_project_root() / "logs" / "langchain_cadv.log")
    logger.info(f"Model: {model_name}")

    original_data_path = get_project_root() / "data" / f"{dataset_name}"

    if downstream_task in ["ml_inference_classification", "ml_inference_regression"]:
        spark_train_df, spark_train, _, _ = load_train_and_test_spark_data(
            dataset_name=dataset_name,
            downstream_task=downstream_task,
            processed_data_label=processed_data_label,
            dq_manager=dq_manager
        )
    elif downstream_task in ["sql_query", "webpage_generation"]:
        spark_train_df, spark_train, _, _ = load_previous_and_new_spark_data(
            dataset_name=dataset_name,
            downstream_task=downstream_task,
            processed_data_label=processed_data_label,
            dq_manager=dq_manager
        )
    else:
        raise ValueError(f"Invalid downstream task: {downstream_task}")
    spark_validation, spark_validation_df = spark_train, spark_train_df

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_train_df, spark_train)

    if downstream_task in ["ml_inference_classification", "ml_inference_regression"]:
        scripts_path_dir = original_data_path / "scripts" / "ml_inference"
    elif downstream_task in ["sql_query", "webpage_generation"]:
        scripts_path_dir = original_data_path / "scripts" / downstream_task
    else:
        raise ValueError(f"Invalid downstream task: {downstream_task}")

    for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
        if len(single_script_name) != 0 and single_script_name != script_path.stem:
            continue
        if script_path.stem.startswith("classification") and downstream_task == "ml_inference_regression":
            continue
        if script_path.stem.startswith("regression") and downstream_task == "ml_inference_classification":
            continue
        processed_data_path = get_project_root() / "data_processed" / dataset_name / downstream_task / f"{processed_data_label}"
        constraints_result_path = processed_data_path / "constraints" / f"{script_path.stem}" / f"tadv_constraints__{model_name}__{assumption_generation_trick}.yaml"
        constraints_result_path.parent.mkdir(parents=True, exist_ok=True)
        relevant_columns_result_path = processed_data_path / "relevant_columns" / f"{script_path.stem}" / f"relevant_columns__{model_name}.txt"
        relevant_columns_result_path.parent.mkdir(parents=True, exist_ok=True)
        task_instance = get_task_instance(script_path)

        if downstream_task in ["ml_inference_classification", "ml_inference_regression"]:
            downstream_task_description = ML_INFERENCE_TASK_DESCRIPTION
        elif downstream_task == "sql_query":
            downstream_task_description = SQL_QUERY_TASK_DESCRIPTION
        elif downstream_task == "webpage_generation":
            downstream_task_description = WEB_TASK_DESCRIPTION
        else:
            raise ValueError(f"Invalid downstream task: {downstream_task}")
        lc = LangChainCADV(model_name=model_name, downstream_task_description=downstream_task_description,
                           assumption_generation_trick=assumption_generation_trick, logger=logger)

        if assumption_generation_trick == "with_deequ":
            deequ_assumptions = dq_manager.get_constraints_for_spark_df(spark_train, spark_train_df).to_string()
            input_variables = {
                "column_desc": column_desc,
                "script": task_instance.original_script,
                "deequ_assumptions": deequ_assumptions,
            }
        else:
            input_variables = {
                "column_desc": column_desc,
                "script": task_instance.original_script,
            }

        relevant_columns_list, expectations, suggestions = lc.invoke(
            input_variables=input_variables, num_stages=3, max_retries=5
        )

        code_list_for_constraints = [item for v in suggestions.values() for item in v]

        # Validate the constraints on the original data to see if they are grammarly correct
        code_list_for_constraints_valid = dq_manager.filter_constraints(code_list_for_constraints, spark_validation,
                                                                        spark_validation_df)
        constraints = Constraints.from_llm_output(relevant_columns_list, expectations, suggestions,
                                                  code_list_for_constraints_valid)

        with open(relevant_columns_result_path, "w") as f:
            f.write("\n".join(relevant_columns_list))
        logger.info(f"Saved relevant columns to {relevant_columns_result_path}")
        constraints.save_to_yaml(constraints_result_path)
        logger.info(f"Saved constraints to {constraints_result_path}")

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()


if __name__ == "__main__":
    import argparse

    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                               "webpage_generation"]
    assumption_generation_trick_options = [None, "with_deequ", "with_experience"]
    model_name_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4.5-preview"]


    def parse_multiple_indices(input_str, options_list):
        """Parses comma-separated indices or 'all'."""
        if input_str.lower() == "all":
            return options_list
        indices = list(map(int, input_str.split(",")))
        return [options_list[i] for i in indices]


    parser = argparse.ArgumentParser(description='Run LangChain CADV')
    parser.add_argument('--dataset-option', type=str, default="all",
                        help='Dataset name. Options: 0: playground-series-s4e10, 1: healthcare_dataset')
    parser.add_argument('--downstream-task-option', type=str, default="all",
                        help='Downstream task. Options: 0: ml_inference_classification, 1: ml_inference_regression, 2: sql_query, 3: webpage_generation')
    parser.add_argument('--model-name-option', type=str, default="1",
                        help='Model name. Options: 0: gpt-3.5-turbo, 1: gpt-4o, 2: gpt-4.5-preview')
    parser.add_argument('--processed-data-label', type=str, default="0",
                        help='Version Label of the processed data')
    parser.add_argument('--assumption-generation-trick-option', type=str, default="all",
                        help='Assumption generation trick. Options: 0: None, 1: with_deequ, 2: with_experience')
    parser.add_argument('--single-script-name', type=str, default="",
                        help='Single script name to run LangChain CADV on. Leave empty to run on all scripts')

    args = parser.parse_args()
    for dataset_name in parse_multiple_indices(args.dataset_option, dataset_name_options):
        for downstream_task in parse_multiple_indices(args.downstream_task_option, downstream_task_options):
            for model_name in parse_multiple_indices(args.model_name_option, model_name_options):
                for assumption_generation_trick in parse_multiple_indices(args.assumption_generation_trick_option,
                                                                          assumption_generation_trick_options):
                    run_langchain_cadv(dataset_name=dataset_name,
                                       downstream_task=downstream_task,
                                       model_name=model_name,
                                       processed_data_label=args.processed_data_label,
                                       assumption_generation_trick=assumption_generation_trick,
                                       single_script_name=args.single_script_name)
