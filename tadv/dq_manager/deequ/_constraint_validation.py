# from pydeequ.checks import *
import warnings

from pydeequ import CheckLevel
from pydeequ.verification import *

warnings.filterwarnings("ignore", message="DataFrame constructor is internal.*")


def apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints):
    final_check_result = []
    for check_str in code_list_for_constraints:
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
