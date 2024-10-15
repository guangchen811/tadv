from cadv_exploration.loader import load_csv, load_py_files
from cadv_exploration.utils import get_project_root
from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.llm.langchain._model import LangChainCADV
from cadv_exploration.deequ import spark_df_from_pandas_df
from cadv_exploration.llm._tasks import DVTask


def prepare_chain_and_data():
    project_root = get_project_root()
    file_path = (
        project_root
        / "data"
        / "prasad22"
        / "healthcare-dataset"
        / "files"
        / "healthcare_dataset.csv"
    )

    df = load_csv(file_path)
    spark_df, spark = spark_df_from_pandas_df(df)
    column_desc = spark_df_to_column_desc(spark, spark_df)

    dir_path = project_root / "data" / "prasad22" / "healthcare-dataset" / "kernels_py"
    scripts = load_py_files(dir_path)

    lang_chain = LangChainCADV()
    relevent_column_target_chain = lang_chain._build_single_chain(
        DVTask.RELEVENT_COLUMN_TARGET
    )
    expectation_extraction_chain = lang_chain._build_single_chain(
        DVTask.EXPECTATION_EXTRACTION
    )

    return (
        column_desc,
        scripts,
        relevent_column_target_chain,
        expectation_extraction_chain,
        spark,
        spark_df,
    )
