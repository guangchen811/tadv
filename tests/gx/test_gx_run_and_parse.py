from tadv.data_models import Constraints
from tadv.dq_manager import GreatExpectationsDataQualityManager
from tadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from tadv.llm.langchain import LangChainTADVGreatExpectationsDialect
from tadv.llm.langchain.prompts.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from tadv.loader import FileLoader
from tadv.utils import load_dotenv, get_project_root


def test_gx_build_single_chain():
    load_dotenv()

    lc = LangChainTADVGreatExpectationsDialect(model_name="gpt-4o-mini",
                                               downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION, )

    dq_manager = GreatExpectationsDataQualityManager()
    train_file_path = get_project_root() / "data" / "toy_example" / "files" / "hospitalisations_train.csv"
    train_data = FileLoader.load_csv(train_file_path, na_values=["NULL"])
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)

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

    accessed_columns_list, expectations, suggestions = lc.invoke(
        input_variables={"column_desc": column_desc, "script": context},
        num_stages=3,
        max_retries=3,
    )
    code_list_for_constraints = [item for v in suggestions.values() for item in v]

    # Validate the constraints on the original data to see if they are grammarly correct
    code_list_for_constraints_valid = dq_manager.filter_valid_constraints_on_spark(code_list_for_constraints,
                                                                                   spark_train, spark_train_data)
    print("Valid constraints:")
    print(code_list_for_constraints_valid)
    constraints = Constraints.from_llm_output(accessed_columns_list, expectations, suggestions,
                                              code_list_for_constraints_valid)
    print("Constraints:")
    print(constraints)
    valid_code_column_map = constraints.get_suggestions_code_column_map(valid_only=True)
    code_list_for_constraints = [item for item in valid_code_column_map.keys()]
    # Validate the constraints on the clean data
    status_on_clean_test_data = dq_manager.validate_on_spark_df(spark_train, spark_train_data,
                                                                code_list_for_constraints)
    validation_results_on_clean_test_data = dq_manager.build_validation_results(code_list_for_constraints,
                                                                                status_on_clean_test_data,
                                                                                valid_code_column_map)
    print(validation_results_on_clean_test_data)
