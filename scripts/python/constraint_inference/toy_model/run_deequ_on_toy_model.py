from cadv_exploration.utils import load_dotenv
from llm.langchain.downstream_task_prompt import CD_TASK_DESCRIPTION

load_dotenv()
from inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from llm.langchain import LangChainCADV
from data_models import Constraints
from scripts.python.utils import setup_logger

from loader import FileLoader

from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.utils import get_project_root


def main():
    logger = setup_logger("toy_example.log")
    result_path = "toy_example_deequ_constraints.yaml"
    dq_manager = DeequDataQualityManager()
    train_file_path = get_project_root() / "data" / "toy_example" / "files" / "hospitalisations_train.csv"
    test_file_path = get_project_root() / "data" / "toy_example" / "files" / "hospitalisations_test.csv"
    train_data = FileLoader.load_csv(train_file_path, na_values=["NULL"])
    test_data = FileLoader.load_csv(test_file_path, na_values=["NULL"])
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = dq_manager.spark_df_from_pandas_df(test_data)

    suggestion = dq_manager.get_suggestion_for_spark_df(spark_train, spark_train_data)
    code_list_for_constraints = [item["code_for_constraint"] for item in suggestion]
    code_list_for_constraints_valid = filter_constraints(code_list_for_constraints, spark_validation,
                                                         spark_validation_data, logger)

    constraints = Constraints.from_deequ_output(suggestion, code_list_for_constraints_valid)
    constraints.save_to_yaml(result_path)

    code_column_map = constraints.get_suggestions_code_column_map(valid_only=False)
    code_list = [item for item in code_column_map.keys()]
    spark_test_data, spark_test = dq_manager.spark_df_from_pandas_df(test_data)
    status_on_test_data = dq_manager.validate_on_spark_df(spark_test, spark_test_data, code_list,
                                                          return_raw=True)
    result_readable = "\n".join([str(list(record)[3:]) for record in status_on_test_data])
    with open("toy_example_deequ_result.txt", "w") as f:
        f.write(result_readable)

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_train_data, spark_train)
    context = """
nonsensitive_df = duckdb.sql("SELECT * EXCLUDE ssn, gender, race
FROM 's3://datalake/latest/hospitalisations.csv'").df()
hosp_df = nonsensitive_df.dropna()
strokes_total = duckdb.sql("SELECT COUNT(*) FROM hosp_df
WHERE diagnosis = 'stroke'").fetch()
strokes_for_rare_bloodtypes = duckdb.sql("SELECT COUNT(*)
FROM hosp_df WHERE diagnosis = 'stroke'
AND bloodtype IN ('AB negative', 'B negative')").fetch()
generate_report(strokes_total, strokes_for_rare_bloodtypes)"""
    result_path_cadv = "toy_example_cadv_constraints.yaml"
    lc = LangChainCADV(model_name="gpt-4o", downstream_task_description=CD_TASK_DESCRIPTION)

    relevant_columns_list, expectations, suggestions = lc.invoke(
        input_variables={"column_desc": column_desc, "script": context},
        num_stages=3,
        max_retries=3
    )
    code_list_for_constraints = [item for v in suggestions.values() for item in v]

    # Validate the constraints on the original data to see if they are grammarly correct
    code_list_for_constraints_valid = dq_manager.filter_constraints(code_list_for_constraints, spark_validation,
                                                                    spark_validation_data, logger)
    constraints = Constraints.from_llm_output(relevant_columns_list, expectations, suggestions,
                                              code_list_for_constraints_valid)

    constraints.save_to_yaml(result_path_cadv)
    logger.info(f"Saved constraints to {result_path}")

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_validation.stop()


if __name__ == "__main__":
    main()
