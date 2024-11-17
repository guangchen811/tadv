from cadv_exploration.utils import load_dotenv

load_dotenv()

from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root


def test_runnable(deequ_wrapper):
    script_id = 0

    project_root = get_project_root()
    file_path = (
            project_root
            / "data"
            / "prasad22"
            / "healthcare-dataset"
            / "files"
            / "healthcare_dataset.csv"
    )
    df = FileLoader.load_csv(file_path)
    spark_df, spark = deequ_wrapper.spark_df_from_pandas_df(df)
    column_desc = spark_df_to_column_desc(spark_df, spark)

    dir_path = project_root / "data" / "prasad22" / "healthcare-dataset" / "kernels_py"
    scripts = FileLoader.load_py_files(dir_path)

    input_variables = {
        "column_desc": column_desc,
        "script": scripts[script_id],
    }
    lc = LangChainCADV(model="gpt-4o-mini")

    relevant_columns_list, expectations, rules = lc.invoke(
        input_variables=input_variables
    )
