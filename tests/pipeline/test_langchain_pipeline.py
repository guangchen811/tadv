from unittest.mock import Mock

from cadv_exploration.utils import load_dotenv

load_dotenv()

from cadv_exploration.llm.langchain.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from cadv_exploration.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.loader import FileLoader


def test_runnable(dq_manager, resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    df = FileLoader.load_csv(file_path)
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    column_desc = DeequInspectorManager().spark_df_to_column_desc(spark_df, spark)

    dir_path = resources_path / "example_dataset_1" / "kernel_py"
    scripts = FileLoader.load_py_files(dir_path)

    input_variables = {
        "column_desc": column_desc,
        "script": scripts[0],
    }
    lc = LangChainCADV(model_name="gpt-4o-mini", downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION,
                       assumption_generation_trick='add_experience', logger=Mock())

    relevant_columns_list, expectations, rules = lc.invoke(
        input_variables=input_variables
    )
