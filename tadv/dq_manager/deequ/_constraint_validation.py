# from pydeequ.checks import *
import warnings

from pydeequ import CheckLevel
from pydeequ.verification import *

warnings.filterwarnings("ignore", message="DataFrame constructor is internal.*")


def validate_suggestions(spark, spark_df, check):
    check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
    check_result_df = VerificationResult.checkResultsAsDataFrame(spark, check_result)
    return check_result_df


def apply_checks_from_strings(spark, spark_df, check_strings):
    final_check_result = []
    for check_str in check_strings:
        check_result = single_check(spark, spark_df, check_str)
        final_check_result.append(check_result)
    return final_check_result


def single_check(spark, spark_df, check_str):
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


def validate_on_df(spark, spark_df, code_list_for_constraints, return_raw):
    check_result_on_post_corruption_df = apply_checks_from_strings(spark, spark_df, code_list_for_constraints)
    if return_raw:
        return check_result_on_post_corruption_df
    status_on_post_corruption_df = [item['constraint_status'] if
                                    item is not None else None for item in check_result_on_post_corruption_df]
    return status_on_post_corruption_df
