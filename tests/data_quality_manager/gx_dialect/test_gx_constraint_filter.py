import pandas as pd

from tadv.dq_manager import GreatExpectationsDataQualityManager


def test_gx_constraint_filter():
    dq_manager = GreatExpectationsDataQualityManager()
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    code_list_for_constraints = [
        'ExpectColumnValuesToNotBeNull(column="a")',
        'ExpectCompoundColumnsToBeUnique(column_list=["b", "c"])',
        'ExpectColumnValuesToBeInSet(column="a", value_set=["foo", "bar", "baz"])',
        'ExpectColumnToExist(column="def")',
        'ExpectColumnValuesToNotBeNull(column="c")',
    ]
    code_list_for_constraints_valid = dq_manager.filter_valid_constraints_on_spark(code_list_for_constraints, spark,
                                                                                   spark_df)
    assert len(code_list_for_constraints_valid) == 3