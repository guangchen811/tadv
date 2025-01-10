from cadv_exploration.utils import load_dotenv

load_dotenv()

import pandas as pd
from pydeequ.checks import *
from pydeequ.verification import *

from cadv_exploration.dq_manager import DeequDataQualityManager


def test_constraint_validation(dq_manager):
    dq_manager = DeequDataQualityManager()
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

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
    result_df = dq_manager.validate_suggestions(spark, spark_df, check)
    assert result_df.collect()[0]["constraint_status"] == "Success"

    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()
