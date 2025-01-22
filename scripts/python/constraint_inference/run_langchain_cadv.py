from cadv_exploration.utils import load_dotenv

load_dotenv()
from cadv_exploration.utils import get_task_instance

from cadv_exploration.data_models import Constraints

from cadv_exploration.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.utils import get_project_root
from scripts.python.utils import setup_logger, load_train_and_test_spark_data
from cadv_exploration.llm.langchain.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION, \
    SQL_QUERY_TASK_DESCRIPTION


def run_langchain_cadv(data_name, model_name, processed_data_idx, assumption_generation_trick, script_name=None):
    dq_manager = DeequDataQualityManager()
    logger = setup_logger("./langchain_cadv.log")
    logger.info(f"Model: {model_name}")

    original_data_path = get_project_root() / "data" / f"{data_name}"
    processed_data_path = get_project_root() / "data_processed" / f"{data_name}" / f"{processed_data_idx}"

    spark_train_df, spark_train, spark_validation_df, spark_validation = load_train_and_test_spark_data(
        data_name=data_name, processed_data_idx=processed_data_idx, dq_manager=dq_manager
    )

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_train_df, spark_train)

    for group_name in ["sql", "ml"]:
        scripts_path_dir = original_data_path / "scripts" / group_name
        for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
            if script_name is not None and script_name != script_path.stem:
                continue

            constraints_result_path = processed_data_path / "constraints" / f"{script_path.stem}" / "cadv_constraints.yaml"
            constraints_result_path.parent.mkdir(parents=True, exist_ok=True)
            relevant_columns_result_path = processed_data_path / "relevant_columns" / f"{script_path.stem}" / "relevant_columns.txt"
            relevant_columns_result_path.parent.mkdir(parents=True, exist_ok=True)
            task_instance = get_task_instance(script_path)

            task_type = script_path.stem.rsplit("_", 1)[0]
            task_group = task_group_mapping(task_type)
            if task_group == 'sql':
                lc = LangChainCADV(model_name=model_name, downstream_task_description=SQL_QUERY_TASK_DESCRIPTION,
                                   assumption_generation_trick=assumption_generation_trick, logger=logger)
            elif task_group == 'ml':
                lc = LangChainCADV(model_name=model_name, downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION,
                                   assumption_generation_trick=assumption_generation_trick, logger=logger)
            else:
                raise ValueError(f"Invalid task group: {task_group}")

            if assumption_generation_trick == "with_deequ":
                deequ_assumptions = dq_manager.get_constraints_for_spark_df(spark_train, spark_train_df).to_string()
                input_variables = {
                    "column_desc": column_desc,
                    "script": task_instance.original_code,
                    "deequ_assumptions": deequ_assumptions,
                }
            else:
                input_variables = {
                    "column_desc": column_desc,
                    "script": task_instance.original_code,
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


def task_group_mapping(task_type):
    return {
        'bi': 'sql',
        'dev': 'sql',
        'exclude_clause': 'sql',
        'feature_engineering': 'sql',
        'classification': 'ml',
        'regression': 'ml'
    }[task_type]


if __name__ == "__main__":
    # data_name = "healthcare_dataset"
    data_name = "playground-series-s4e10"
    model_name = "gpt-4o"
    run_langchain_cadv(data_name=data_name, model_name=model_name, processed_data_idx='base_version',
                       assumption_generation_trick=None)
    run_langchain_cadv(data_name=data_name, model_name=model_name, processed_data_idx="with_deequ",
                       assumption_generation_trick="with_deequ")
    run_langchain_cadv(data_name=data_name, model_name=model_name, processed_data_idx="with_experience",
                       assumption_generation_trick="with_experience")
