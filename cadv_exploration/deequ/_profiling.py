from typing import List

import pandas as pd
from pydeequ.profiles import *


def profile_on_spark_df(spark, spark_df) -> pd.DataFrame:
    result = ColumnProfilerRunner(spark).onData(spark_df).run()
    return result
