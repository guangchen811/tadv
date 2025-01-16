from abc import ABC, abstractmethod


class AbstractDataQualityManager(ABC):
    """
    Abstract base class for managing data quality operations using different backends.
    """

    @abstractmethod
    def spark_df_from_pandas_df(self, pandas_df):
        """
        Convert a pandas DataFrame to a Spark DataFrame.
        """
        pass

    @abstractmethod
    def analyze_on_spark_df(self, spark, spark_df, analyzers):
        """
        Perform data analysis on a Spark DataFrame.
        """
        pass

    @abstractmethod
    def profile_on_spark_df(self, spark, spark_df):
        """
        Profile the data in a Spark DataFrame.
        """
        pass

    @abstractmethod
    def get_suggestion_for_spark_df(self, spark, spark_df):
        """
        Generate suggestions for improving data quality for a Spark DataFrame.
        """
        pass

    @abstractmethod
    def validate_suggestions(self, spark, spark_df, check):
        """
        Validate suggestions on a Spark DataFrame.
        """
        pass

    @abstractmethod
    def apply_checks_from_strings(self, spark, spark_df, check_strings):
        """
        Apply validation checks provided as strings to a Spark DataFrame.
        """
        pass

    @abstractmethod
    def validate_on_spark_df(self, spark, spark_df, code_list_for_constraints, return_raw=False):
        """
        Validate a Spark DataFrame against specified constraints.
        """
        pass

    @abstractmethod
    def filter_constraints(self, code_list_for_constraints, spark_original_validation, spark_original_validation_df,
                           logger):
        """
        Filter constraints based on validation results.
        """
        pass
