from cadv_exploration.dq_manager.abstract_data_quality_manager import AbstractDataQualityManager
from cadv_exploration.dq_manager.deequ._analyzing import analyze_on_spark_df
from cadv_exploration.dq_manager.deequ._constraint_suggestion import \
    get_suggestion_for_spark_df
from cadv_exploration.dq_manager.deequ._constraint_validation import validate_suggestions, apply_checks_from_strings, \
    validate_on_df
from cadv_exploration.dq_manager.deequ._loading import spark_df_from_pandas_df
from cadv_exploration.dq_manager.deequ._profiling import profile_on_spark_df


class DeequDataQualityManager(AbstractDataQualityManager):
    def __init__(self):
        super().__init__()

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

    def validate_on_spark_df(self, spark, spark_df, code_list_for_constraints, return_raw=False):
        return validate_on_df(spark, spark_df, code_list_for_constraints, return_raw)

    def filter_constraints(self, code_list_for_constraints, spark_original_validation, spark_original_validation_df,
                           logger):
        logger.info(f"Suggested Code list for constraints: {code_list_for_constraints}")
        check_result_on_original_validation_df = self.apply_checks_from_strings(spark_original_validation,
                                                                                spark_original_validation_df,
                                                                                code_list_for_constraints)
        status_on_original_validation_df = [item['constraint_status'] if
                                            item is not None else None for item in
                                            check_result_on_original_validation_df]
        success_on_original_validation_df = status_on_original_validation_df.count("Success")
        failure_check_on_original_validation_df = [code_list_for_constraints[i] for i in
                                                   range(len(check_result_on_original_validation_df)) if
                                                   check_result_on_original_validation_df[i] is not None and
                                                   check_result_on_original_validation_df[i][
                                                       'constraint_status'] == 'Failure']
        failure_check_output_on_original_validation_df = "\n".join(failure_check_on_original_validation_df)
        failure_check_on_original_validation_df = [code_list_for_constraints[i] for i in
                                                   range(len(code_list_for_constraints)) if
                                                   check_result_on_original_validation_df[i] is None]
        grammarly_failure_check_output_on_original_validation_df = "\n".join(failure_check_on_original_validation_df)
        logger.info(f"Check result on original data: {check_result_on_original_validation_df}")
        logger.info(
            f"Success on original data: {success_on_original_validation_df} / {len(status_on_original_validation_df)}")
        logger.info(f"Failure check on original data: {failure_check_output_on_original_validation_df}")
        logger.info(
            f"Grammarly failure check on original data: {grammarly_failure_check_output_on_original_validation_df}")
        # remove the constraints that are not grammarly correct
        code_list_for_constraints = [code_list_for_constraints[i] for i in range(len(code_list_for_constraints)) if
                                     status_on_original_validation_df[i] == "Success"]
        logger.info(f"Filtered Code list for constraints: {code_list_for_constraints}")
        return code_list_for_constraints
