from abc import ABC

import numpy as np
import pydeequ
from pyspark.sql import SparkSession


class AbstractDataQualityManager(ABC):
    """
    Abstract base class for managing data quality operations using different backends.
    """

    def spark_df_from_pandas_df(self, pandas_df):
        spark = (
            SparkSession.builder
            .config("spark.jars.ivy.log", "none")
            .config("spark.hadoop.native.lib", "false")
            .config("spark.jars.packages", pydeequ.deequ_maven_coord)
            .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
            .config("spark.driver.host", "localhost")
            .getOrCreate()
        )
        pandas_df = pandas_df.where(pandas_df.notna(), None)
        pandas_df = pandas_df.replace(np.nan, None)
        spark_df = spark.createDataFrame(pandas_df)
        return spark_df, spark
