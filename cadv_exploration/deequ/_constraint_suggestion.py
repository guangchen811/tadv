from pydeequ.suggestions import *


def get_suggestion_for_spark_df(spark, spark_df):
    result = (
        ConstraintSuggestionRunner(spark)
        .onData(spark_df)
        .addConstraintRule(DEFAULT())
        .run()
    )
    return result
