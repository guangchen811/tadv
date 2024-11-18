from cadv_exploration.deequ_wrapper._analyzing import analyze_on_spark_df
from cadv_exploration.deequ_wrapper._constraint_suggestion import \
    get_suggestion_for_spark_df
from cadv_exploration.deequ_wrapper._constraint_validation import validate_suggestions, apply_checks_from_strings, \
    validate_on_df
from cadv_exploration.deequ_wrapper._loading import spark_df_from_pandas_df
from cadv_exploration.deequ_wrapper._profiling import profile_on_spark_df


class DeequWrapper():
    def __init__(self):
        pass

    def spark_df_from_pandas_df(self, pandas_df):
        return spark_df_from_pandas_df(pandas_df)

    def analyze_on_spark_df(self, spark, spark_df, analyzers):
        return analyze_on_spark_df(spark, spark_df, analyzers)

    def profile_on_spark_df(self, spark, spark_df):
        return profile_on_spark_df(spark, spark_df)

    def get_suggestion_for_spark_df(self, spark, spark_df):
        return get_suggestion_for_spark_df(spark, spark_df)

    def validate_suggestions(self, spark, spark_df, check):
        return validate_suggestions(spark, spark_df, check)

    def apply_checks_from_strings(self, spark, spark_df, check_strings):
        return apply_checks_from_strings(spark, spark_df, check_strings)

    def validate_on_df(self, spark, spark_df, code_list_for_constraints):
        return validate_on_df(spark, spark_df, code_list_for_constraints)
