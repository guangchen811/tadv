from cadv_exploration.utils import load_dotenv

load_dotenv()

from inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from llm.langchain import LangChainCADV
from llm.langchain.downstream_task_prompt import CD_TASK_DESCRIPTION
from loader import FileLoader

from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.utils import get_project_root


def main():
    dq_manager = DeequDataQualityManager()
    train_file_path = get_project_root() / "data" / "toy_example" / "files" / "hospitalisations_train.csv"
    train_data = FileLoader.load_csv(train_file_path)
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    suggestion = dq_manager.get_suggestion_for_spark_df(spark_train, spark_train_data)
    print(suggestion)

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
    lc = LangChainCADV(model_name="llama3.2", downstream_task_description=CD_TASK_DESCRIPTION)

    relevant_columns_list, expectations, suggestions = lc.invoke(
        input_variables={"column_desc": column_desc, "script": context},
        num_stages=3,
        max_retries=3
    )

    print(suggestions)

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()


if __name__ == "__main__":
    main()
