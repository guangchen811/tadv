from typing import List

import pandas as pd
from pydeequ.analyzers import AnalysisRunner, AnalyzerContext


def analyze_on_spark_df(spark, spark_df, analyzers: List) -> pd.DataFrame:
    runner = AnalysisRunner(spark).onData(spark_df)
    for analyzer in analyzers:
        runner = runner.addAnalyzer(analyzer)
    analysis_result = runner.run()
    analysis_result_df = AnalyzerContext.successMetricsAsDataFrame(spark, analysis_result)
    pandas_df = analysis_result_df.toPandas()
    return pandas_df
