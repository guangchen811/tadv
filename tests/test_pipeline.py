import os

os.environ["SPARK_VERSION"] = "3.5"

from cadv_exploration.pipeline import prepare_chain_and_data


def test_prepare_chain_and_data():
    (
        column_desc,
        scripts,
        relevent_column_target_chain,
        expectation_extraction_chain,
        spark,
        spark_df,
    ) = prepare_chain_and_data()
    assert relevent_column_target_chain.first.input_variables == [
        "code_snippet",
        "columns",
    ]
    assert expectation_extraction_chain.first.input_variables == [
        "code_snippet",
        "columns",
        "relevant_columns",
    ]
    relevant_columns_list = relevent_column_target_chain.invoke(
        {"code_snippet": scripts[0], "columns": column_desc}
    )
    assert isinstance(relevant_columns_list, list)
    assert len(relevant_columns_list) > 0
    expectations = expectation_extraction_chain.invoke(
        {
            "code_snippet": scripts[0],
            "columns": column_desc,
            "relevant_columns": str(relevant_columns_list),
        }
    )
    assert isinstance(expectations, dict)
