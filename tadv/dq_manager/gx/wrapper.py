from tadv.data_models import ValidationResults
from tadv.dq_manager.abstract_data_quality_manager import AbstractDataQualityManager
from tadv.dq_manager.gx._constraint_validation import apply_checks_from_strings_on_spark_df


class GreatExpectationsDataQualityManager(AbstractDataQualityManager):
    def __init__(self):
        super().__init__()

    @staticmethod
    def apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints):
        return apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints)

    def validate_on_spark_df(self, spark, spark_df, code_list_for_constraints, return_raw=False):
        check_result = self.apply_checks_from_strings_on_spark_df(spark, spark_df,
                                                                  code_list_for_constraints)
        if return_raw:
            return check_result
        status = check_result["results"]
        return status

    def build_validation_results(self, code_list_for_constraints, status_on_clean_test_data,
                                 valid_code_column_map) -> ValidationResults:
        raise NotImplementedError(
            "Implement this method after finishing the implementation of suggestions generation process.")
