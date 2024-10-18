# from pydeequ.checks import *
from pydeequ.verification import *


def validate_suggestions(spark, spark_df, check):
    checkResult = VerificationSuite(spark).onData(spark_df).addCheck(check).run()
    checkResult_df = VerificationResult.checkResultsAsDataFrame(spark, checkResult)
    return checkResult_df
