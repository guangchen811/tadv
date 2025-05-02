from tadv.dq_manager import GreatExpectationsDataQualityManager
from tadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from tadv.llm.langchain.models.sequential_gx_model_with_scope import SequentialLangChainTADVGreatExpectationsDialect
from tadv.llm.langchain.prompts.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from tadv.loader import FileLoader
from tadv.utils import load_dotenv, get_project_root


def test_gx_build_single_chain():
    load_dotenv()
    lc = SequentialLangChainTADVGreatExpectationsDialect(model_name="gpt-4o-mini",
                                                         downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION, )
    dq_manager = GreatExpectationsDataQualityManager()
    train_file_path = get_project_root() / "data" / "toy_example" / "files" / "hospitalisations_train.csv"
    train_data = FileLoader.load_csv(train_file_path, na_values=["NULL"])
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)

    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_train_data, spark_train)
    context = """nonsensitive_df = duckdb.sql("SELECT * EXCLUDE ssn, gender, race
    FROM 's3://datalake/latest/hospitalisations.csv'").df()
    hosp_df = nonsensitive_df.dropna()
    strokes_total = duckdb.sql("SELECT COUNT(*) FROM hosp_df
    WHERE diagnosis = 'stroke'").fetch()
    strokes_for_rare_bloodtypes = duckdb.sql("SELECT COUNT(*)
    FROM hosp_df WHERE diagnosis = 'stroke'
    AND bloodtype IN ('AB negative', 'B negative')").fetch()
    generate_report(strokes_total, strokes_for_rare_bloodtypes)"""

    prompts = lc.show_prompts(
        input_variables={"column_desc": column_desc, "script": context}, num_stages=3)
    for k, v in prompts.items():
        print(f"Stage {k}:")
        print(v)
        print("\n")
