from abc import ABC, abstractmethod

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

    @abstractmethod
    def analyze_on_spark_df(self, spark, spark_df, analyzers):
        """
        Perform data analysis on a Spark DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def profile_on_spark_df(self, spark, spark_df):
        """
        Profile the data in a Spark DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def get_suggestion_for_spark_df(self, spark, spark_df):
        """
        Generate suggestions for improving data quality for a Spark DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_suggestions(self, spark, spark_df, check):
        """
        Validate suggestions on a Spark DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_checks_from_strings(self, spark, spark_df, check_strings):
        """
        Apply validation checks provided as strings to a Spark DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_on_spark_df(self, spark, spark_df, code_list_for_constraints, return_raw=False):
        """
        Validate a Spark DataFrame against specified constraints.
        """
        raise NotImplementedError

    @abstractmethod
    def filter_constraints(self, code_list_for_constraints, spark_original_validation, spark_original_validation_df):
        """
        Filter constraints based on validation results.
        """
        raise NotImplementedError
