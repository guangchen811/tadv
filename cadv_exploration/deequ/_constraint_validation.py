# from pydeequ.checks import *
from pydeequ import CheckLevel
from pydeequ.verification import *


def validate_suggestions(spark, spark_df, check):
    check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
    check_result_df = VerificationResult.checkResultsAsDataFrame(spark, check_result)
    return check_result_df


def apply_checks_from_strings(check, check_strings, spark, spark_df):
    final_check_result = []
    for check_str in check_strings:
        check_result = single_check(check_str, spark, spark_df)
        final_check_result.append(check_result)
    return final_check_result


def single_check(check_str, spark, spark_df):
    check = Check(spark, CheckLevel.Warning, "Check for data")
    try:
        if check_str.startswith("."):
            exec(f"check.addConstraint(check{check_str})")
        else:
            exec(f"check.addConstraint(check.{check_str})")
        check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
        check_result = VerificationResult.checkResultsAsDataFrame(
            spark, check_result
        ).collect()[0]
    except Exception as e:
        check_result = None
    return check_result

# def apply_checks_from_strings(check, check_strings, spark, spark_df):
#     for check_str in check_strings:
#         exec(f"check.addConstraint(check.{check_str})")
#     check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
#     check_result = VerificationResult.checkResultsAsDataFrame(
#         spark, check_result
#     ).collect()
#     return check_result
