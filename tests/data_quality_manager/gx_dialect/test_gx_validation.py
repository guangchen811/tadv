import time
from pprint import pprint

import pandas as pd

from tadv.dq_manager import GreatExpectationsDataQualityManager


def test_gx_quality_manager():
    dq_manager = GreatExpectationsDataQualityManager()
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    code_list_for_constraints = [
        'ExpectColumnToExist(column="def")',
        'ExpectColumnValuesToNotBeNull(column="a")',
        'ExpectCompoundColumnsToBeUnique(column_list=["b", "c"])',
        'ExpectColumnValuesToBeInSet(column="a", value_set=["foo", "bar", "baz"])',
        'ExpectColumnValuesToNotBeNull(column="c")',
    ]
    start = time.time()
    check_result = dq_manager.validate_on_spark_df(spark, spark_df, code_list_for_constraints)
    end = time.time()
    print(f"Time taken for validation: {end - start} seconds")

    assert check_result[0]["success"] == False
    assert check_result[1]["success"] == True
    assert check_result[2]["success"] == True
    assert check_result[3]["success"] == True
    assert check_result[4]["success"] == False

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()


def test_gx_quality_manager_with_incorrect_grammar():
    dq_manager = GreatExpectationsDataQualityManager()
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    code_list_for_constraints = [
        'xpectColumnToExist(column="def")',  # grammatically incorrect
        'ExpectColumnValueToNotBeNull(column="a")',  # grammatically incorrect
        'ExpectCompoundColumnsToBeUnique(column_list=["b", "c"])',
        'ExpectColumnValuesToBeInSet(column="a", value_set=["foo", "bar", "baz"])',
        'ExpectColumnValuesToNotBeNull(column="c")',
    ]
    start = time.time()
    check_result = dq_manager.validate_on_spark_df(spark, spark_df, code_list_for_constraints)
    end = time.time()
    print(f"Time taken for validation: {end - start} seconds")

    assert check_result[0]["success"] == False  # failed due to incorrect grammar
    assert check_result[1]["success"] == False  # failed due to incorrect grammar
    assert check_result[2]["success"] == True
    assert check_result[3]["success"] == True
    assert check_result[4]["success"] == False
    pprint(check_result)

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()
