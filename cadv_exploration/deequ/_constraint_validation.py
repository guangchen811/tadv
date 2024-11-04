# from pydeequ.checks import *
from pydeequ.verification import *


def validate_suggestions(spark, spark_df, check):
    check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
    check_result_df = VerificationResult.checkResultsAsDataFrame(spark, check_result)
    return check_result_df


def apply_checks_from_strings(check, check_strings, spark, spark_df):
    for check_str in check_strings:
        exec(f"check.addConstraint(check.{check_str})")
    check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
    check_result = VerificationResult.checkResultsAsDataFrame(
        spark, check_result
    ).collect()
    return check_result
