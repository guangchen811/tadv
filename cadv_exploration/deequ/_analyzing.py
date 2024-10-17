from typing import List

import pandas as pd
from pydeequ.analyzers import AnalysisRunner, AnalyzerContext


def analyze_on_spark_df(spark, spark_df, analyzers: List) -> pd.DataFrame:
    runner = AnalysisRunner(spark).onData(spark_df)
    for analyzer in analyzers:
        runner = runner.addAnalyzer(analyzer)
    analysisResult = runner.run()
    analysisResult_df = AnalyzerContext.successMetricsAsDataFrame(spark, analysisResult)
    pandas_df = analysisResult_df.toPandas()
    return pandas_df
