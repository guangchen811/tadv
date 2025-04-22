# from pydeequ.checks import *
import warnings

from pydeequ import CheckLevel
from pydeequ.verification import *

warnings.filterwarnings("ignore", message="DataFrame constructor is internal.*")


def apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints, isolated_check):
    """
    Apply checks from strings on a Spark DataFrame.
    :param spark: Spark session.
    :param spark_df: Spark DataFrame to apply checks on.
    :param code_list_for_constraints: List of strings representing the checks to apply.
    :param isolated_check: If True, each check is run in isolation. It is recommended to set this when you are not sure whether all the checks are grammarly correct to avoid all checks failing without result.
    """
    if isolated_check:
        final_check_result = []
        for check_str in code_list_for_constraints:
            check_result = single_check(spark, spark_df, check_str)
            final_check_result.append(check_result)
    else:
        check = Check(spark, CheckLevel.Warning, "Check for data")
        try:
            for check_str in code_list_for_constraints:
                check_str = _normalize_check_str(check_str)
                exec(f"check.addConstraint(check.{check_str})")
            check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
            final_check_result = VerificationResult.checkResultsAsDataFrame(
                spark, check_result
            ).collect()
        except Exception as e:
            final_check_result = [
                {
                    "constraint_status": "Failure",
                    "constraint_message": str(e),
                }
            ]
    return final_check_result


def single_check(spark, spark_df, check_str):
    check = Check(spark, CheckLevel.Warning, "Check for data")
    try:
        check_str = _normalize_check_str(check_str)
        exec(f"check.addConstraint(check.{check_str})")
        check_result = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
        check_result = VerificationResult.checkResultsAsDataFrame(
            spark, check_result
        ).collect()[0]
    except Exception as e:
        check_result = {
            "constraint_status": "Failure",
            "constraint_message": str(e),
            "constraint_code": check_str,
        }
    return check_result


def _normalize_check_str(check_str):
    normalized = check_str if not check_str.startswith(".") else check_str[1:]
    return normalized
