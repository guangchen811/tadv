import time

import pandas as pd
from pydeequ.checks import *
from pydeequ.verification import *


def test_validation_on_small_dataset_with_pydeequ(dq_manager):
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    check = Check(spark, CheckLevel.Warning, "Review Check")

    added_checks = [
        check.hasSize(lambda x: x >= 3),
        check.hasMin("b", lambda x: x == 0),
        check.isComplete("c"),
        check.isUnique("a"),
        check.isContainedIn("a", ["foo", "bar", "baz"]),
        check.isNonNegative("b"),
    ]
    start = time.time()
    for added_check in added_checks:
        check.addConstraint(added_check)
    check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()

    check_result = VerificationResult.checkResultsAsDataFrame(
        spark, check_result
    ).collect()
    end = time.time()

    print(f"Time taken for validation: {end - start} seconds")

    assert check_result[0]["constraint_status"] == "Success"
    assert check_result[1]["constraint_status"] == "Failure"
    assert check_result[2]["constraint_status"] == "Failure"
    assert check_result[3]["constraint_status"] == "Success"
    assert check_result[4]["constraint_status"] == "Success"
    assert check_result[5]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()


def test_validation_on_small_dataset_in_single_list(dq_manager):
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    check_strings = [
        ".hasSize(lambda x: x >= 3)",
        ".hasMin('b', lambda x: x == 0)",
        "isComplete('c')",
        ".isUnique('a')",
        "isContainedIn('a', ['foo', 'bar', 'baz'])",
        ".isNonNegative('b')"
    ]
    start = time.time()
    check_result = dq_manager.apply_checks_from_strings_on_spark_df(spark, spark_df, check_strings,
                                                                    isolated_check=False)
    end = time.time()

    print(f"Time taken for validation: {end - start} seconds")
    assert check_result[0]["constraint_status"] == "Success"
    assert check_result[1]["constraint_status"] == "Failure"
    assert check_result[2]["constraint_status"] == "Failure"
    assert check_result[3]["constraint_status"] == "Success"
    assert check_result[4]["constraint_status"] == "Success"
    assert check_result[5]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()


def test_validation_on_small_dataset_in_single_list_with_incorrect_grammar_and_isolated_check(dq_manager):
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    check_strings = [
        ".hasSize(lambda x: x >= 3)",
        ".hasMin('b', lambda x: x == 0)",
        "isComplete('c')",
        ".isUnique('a')",
        "isCont('a', ['foo', 'bar', 'baz'])",  # grammatically incorrect
        ".isNonNegative('b')"
    ]
    start = time.time()
    check_result = dq_manager.apply_checks_from_strings_on_spark_df(spark, spark_df, check_strings,
                                                                    isolated_check=True)
    end = time.time()

    print(f"Time taken for validation: {end - start} seconds")
    assert check_result[0]["constraint_status"] == "Success"
    assert check_result[1]["constraint_status"] == "Failure"
    assert check_result[2]["constraint_status"] == "Failure"
    assert check_result[3]["constraint_status"] == "Success"
    assert check_result[4]["constraint_status"] == "Failure"  # failure because of grammatically incorrect
    assert check_result[5]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()


def test_validation_on_small_dataset_in_single_list_with_incorrect_grammar_and_isolated_check_false(dq_manager):
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    check_strings = [
        ".hasSize(lambda x: x >= 3)",
        ".hasMin('b', lambda x: x == 0)",
        "isComplete('c')",
        ".isUnique('a')",
        "isCont('a', ['foo', 'bar', 'baz'])",  # grammatically incorrect
        ".isNonNegative('b')"
    ]
    start = time.time()
    check_result = dq_manager.apply_checks_from_strings_on_spark_df(spark, spark_df, check_strings,
                                                                    isolated_check=False)
    end = time.time()

    print(f"Time taken for validation: {end - start} seconds")
    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()
    assert len(check_result) == 1
    assert check_result[0]["constraint_status"] == "Failure"


def test_validation_on_small_dataset_in_single_list_with_isolated_check_false(dq_manager):
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    check_strings = [
        ".hasSize(lambda x: x >= 3)",
        ".hasMin('b', lambda x: x == 0)",
        "isComplete('c')",
        ".isUnique('a')",
        ".isContainedIn('a', ['foo', 'bar', 'baz'])",
        ".isNonNegative('b')"
    ]
    start = time.time()
    check_result = dq_manager.apply_checks_from_strings_on_spark_df(spark, spark_df, check_strings,
                                                                    isolated_check=False)
    end = time.time()

    print(f"Time taken for validation: {end - start} seconds")
    assert check_result[0]["constraint_status"] == "Success"
    assert check_result[1]["constraint_status"] == "Failure"
    assert check_result[2]["constraint_status"] == "Failure"
    assert check_result[3]["constraint_status"] == "Success"
    assert check_result[4]["constraint_status"] == "Success"
    assert check_result[5]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()
