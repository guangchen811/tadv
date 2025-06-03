from abc import ABC, abstractmethod

import numpy as np
import pydeequ
from pyspark.sql import SparkSession


class AbstractDataQualityManager(ABC):
    """
    Abstract base class for managing data quality operations using different backends.
    """

    @staticmethod
    def spark_df_from_pandas_df(pandas_df):
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

    @abstractmethod
    def validate_on_spark_df(self, spark, spark_df, code_list_for_constraints, return_raw=False):
        """
        Validate the Spark DataFrame using the provided constraints.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def filter_valid_constraints_on_spark(self, code_list_for_constraints, spark, spark_df) -> list:
        """
        Filter out invalid constraints from the provided list.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    @abstractmethod
    def build_validation_results(code_list_for_constraints, status, valid_code_column_map):
        """
        Build validation results based on the constraints and their statuses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
