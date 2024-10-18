import pandas as pd
from pydeequ.checks import *
from pydeequ.verification import *

# https://github.com/awslabs/python-deequ/issues/198
# import gc
from cadv_exploration.deequ import validate_suggestions
from cadv_exploration.deequ import spark_df_from_pandas_df


def test_constraint_validation():
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = spark_df_from_pandas_df(df)

    check = Check(spark, CheckLevel.Warning, "Review Check")
    check.hasSize(lambda x: x >= 3)
    added_checks = [
        check.hasSize(lambda x: x >= 3),
        check.hasMin("b", lambda x: x == 0),
        check.isComplete("c"),
        check.isUnique("a"),
        check.isContainedIn("a", ["foo", "bar", "baz"]),
        check.isNonNegative("b"),
    ]
    check.addConstraints(added_checks)
    result_df = validate_suggestions(spark, spark_df, check)
    assert result_df.collect()[0]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()
