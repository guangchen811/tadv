import pandas as pd

from tadv.dq_manager.deequ._constraint_suggestion import get_suggestion_for_spark_df


def test_get_suggestion_for_spark_df(dq_manager):
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    suggestions = get_suggestion_for_spark_df(spark, spark_df)

    assert isinstance(suggestions, list)

    for s in suggestions:
        assert "constraint_name" in s
        assert "column_name" in s or "description" in s
    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()
