from tadv.utils import load_dotenv

load_dotenv()
from workflow.s2_experiments.utils import setup_logger
from tadv.llm.langchain.downstream_task_prompt import SQL_QUERY_TASK_DESCRIPTION
from tadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from tadv.llm.langchain import LangChainTADV

from tadv.loader import FileLoader

from tadv.dq_manager import DeequDataQualityManager
from tadv.utils import get_project_root

logger = setup_logger("sandbox_example.log")

dq_manager = DeequDataQualityManager()
train_file_path = get_project_root() / "data" / "toy_example" / "files" / "hospitalisations_train.csv"
test_file_path = get_project_root() / "data" / "toy_example" / "files" / "hospitalisations_test.csv"
train_data = FileLoader.load_csv(train_file_path, na_values=["NULL"])
test_data = FileLoader.load_csv(test_file_path, na_values=["NULL"])
spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
spark_test_data, spark_test = dq_manager.spark_df_from_pandas_df(test_data)

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

lc = LangChainTADV(model_name="microsoft/Phi-3-mini-4k-instruct",
                   downstream_task_description=SQL_QUERY_TASK_DESCRIPTION,
                   assumption_generation_trick=None, logger=logger)

relevant_columns_list, expectations, suggestions = lc.invoke(
    input_variables={"column_desc": column_desc, "script": context},
    num_stages=3,
    max_retries=3
)
