from cadv_exploration.utils import load_dotenv

load_dotenv()
import gc

import pandas as pd
from pydeequ.checks import *
from pydeequ.verification import *

from cadv_exploration.deequ import spark_df_from_pandas_df

# def test_validation_on_small_dataset():
#     df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
#     spark_df, spark = spark_df_from_pandas_df(df)

#     check = Check(spark, CheckLevel.Warning, "Review Check")

#     added_checks = [
#         check.hasSize(lambda x: x >= 3),
#         # check.hasMin("b", lambda x: x == 0),
#         # check.isComplete("c"),
#         # check.isUnique("a"),
#         # check.isContainedIn("a", ["foo", "bar", "baz"]),
#         # check.isNonNegative("b"),
#     ]

#     for added_check in added_checks:
#         check.addConstraint(added_check)
#     checkResult = VerificationSuite(spark).onData(spark_df).addCheck(check).run()

#     checkResult = VerificationResult.checkResultsAsDataFrame(
#         spark, checkResult
#     ).collect()

#     assert checkResult[0]["constraint_status"] == "Success"
#     assert checkResult[1]["constraint_status"] == "Failure"
#     assert checkResult[2]["constraint_status"] == "Success"
#     assert checkResult[3]["constraint_status"] == "Success"
#     assert checkResult[4]["constraint_status"] == "Success"
#     assert checkResult[5]["constraint_status"] == "Success"
#     # Explicitly release the check object and clean up
#     check = None
#     checkResult = None

#     # Stop the Spark session to release resources and prevent hanging
#     spark.stop()

#     # Force garbage collection (optional)
#     gc.collect()
