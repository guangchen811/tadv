from cadv_exploration.utils import load_dotenv

load_dotenv()

from cadv_exploration.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.utils import get_project_root
from cadv_exploration.data_models import Constraints
from cadv_exploration.llm.langchain.downstream_task_prompt import SQL_QUERY_TASK_DESCRIPTION
from scripts.python.utils import setup_logger, parse_arguments, \
    load_train_and_test_spark_data


def run_langchain_cadv(data_name, processed_data_idx):
    logger = setup_logger("./langchain_cadv.log")
    args = parse_arguments(description="Run LangChain CADV on sql tasks")
    dq_manager = DeequDataQualityManager()

    logger.info(f"Model: {args.model}")

    original_data_path = get_project_root() / "data" / f"{data_name}"
    processed_data_path = get_project_root() / "data_processed" / f"{data_name}" / f"{processed_data_idx}"

    spark_train_data, spark_train, spark_validation_data, spark_validation = load_train_and_test_spark_data(
        data_name=data_name, processed_data_idx=processed_data_idx, dq_manager=dq_manager
    )

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_train_data, spark_train)

    max_retries = args.max_retries
    lc = LangChainCADV(model_name=args.model, downstream_task_description=SQL_QUERY_TASK_DESCRIPTION)

    scripts_path_dir = original_data_path / "scripts_sql"
    for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
        if not script_path.name.endswith(".sql"):
            continue
        result_path = processed_data_path / "constraints" / f"{script_path.name.split('.')[0]}" / "cadv_constraints.yaml"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        script_context = script_path.read_text()

        input_variables = {
            "column_desc": column_desc,
            "script": script_context,
        }

        relevant_columns_list, expectations, suggestions = lc.invoke(
            input_variables=input_variables, num_stages=3, max_retries=max_retries
        )

        code_list_for_constraints = [item for v in suggestions.values() for item in v]

        # Validate the constraints on the original data to see if they are grammarly correct
        code_list_for_constraints_valid = dq_manager.filter_constraints(code_list_for_constraints, spark_validation,
                                                                        spark_validation_data)
        constraints = Constraints.from_llm_output(relevant_columns_list, expectations, suggestions,
                                                  code_list_for_constraints_valid)
        constraints.save_to_yaml(result_path)
        logger.info(f"Saved constraints to {result_path}")

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()


if __name__ == "__main__":
    run_langchain_cadv(data_name="playground-series-s4e10", processed_data_idx=8)
