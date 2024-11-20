from cadv_exploration.utils import load_dotenv

load_dotenv()

from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.loader import FileLoader


def test_runnable(deequ_wrapper, resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    df = FileLoader.load_csv(file_path)
    spark_df, spark = deequ_wrapper.spark_df_from_pandas_df(df)
    column_desc = spark_df_to_column_desc(spark_df, spark)

    dir_path = resources_path / "example_dataset_1" / "kernel_py"
    scripts = FileLoader.load_py_files(dir_path)

    input_variables = {
        "column_desc": column_desc,
        "script": scripts[0],
    }
    lc = LangChainCADV(model="gpt-4o-mini")

    relevant_columns_list, expectations, rules = lc.invoke(
        input_variables=input_variables
    )
