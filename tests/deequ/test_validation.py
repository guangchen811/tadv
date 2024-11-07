from cadv_exploration.utils import load_dotenv

load_dotenv()

import pandas as pd
from pydeequ.checks import *
from pydeequ.verification import *

from cadv_exploration.deequ import spark_df_from_pandas_df, apply_checks_from_strings


def test_validation_on_small_dataset():
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = spark_df_from_pandas_df(df)

    check = Check(spark, CheckLevel.Warning, "Review Check")

    added_checks = [
        check.hasSize(lambda x: x >= 3),
        check.hasMin("b", lambda x: x == 0),
        check.isComplete("c"),
        check.isUnique("a"),
        check.isContainedIn("a", ["foo", "bar", "baz"]),
        check.isNonNegative("b"),
    ]

    for added_check in added_checks:
        check.addConstraint(added_check)
    check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()

    check_result = VerificationResult.checkResultsAsDataFrame(
        spark, check_result
    ).collect()

    assert check_result[0]["constraint_status"] == "Success"
    assert check_result[1]["constraint_status"] == "Failure"
    assert check_result[2]["constraint_status"] == "Success"
    assert check_result[3]["constraint_status"] == "Success"
    assert check_result[4]["constraint_status"] == "Success"
    assert check_result[5]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()


def test_validation_on_small_dataset():
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = spark_df_from_pandas_df(df)

    check = Check(spark, CheckLevel.Warning, "Review Check")

    check_strings = [
        ".hasSize(lambda x: x >= 3)",
        ".hasMin('b', lambda x: x == 0)",
        ".isComplete('c')",
        ".isUnique('a')",
        ".isContainedIn('a', ['foo', 'bar', 'baz'])",
        ".isNonNegative('b')"
    ]

    check_result = apply_checks_from_strings(check, check_strings, spark, spark_df)

    assert check_result[0]["constraint_status"] == "Success"
    assert check_result[1]["constraint_status"] == "Failure"
    assert check_result[2]["constraint_status"] == "Success"
    assert check_result[3]["constraint_status"] == "Success"
    assert check_result[4]["constraint_status"] == "Success"
    assert check_result[5]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()
