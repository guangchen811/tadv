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
        code_status_map = {code_list_for_constraints[i]: status_on_clean_test_data[i] for i in
                           range(len(code_list_for_constraints))}
        validation_results_dict = {"results": {column: {"code": []} for column in valid_code_column_map.values()}}
        for code, column in valid_code_column_map.items():
            validation_results_dict["results"][column]["code"].append(
                [code, "Passed" if code_status_map[code]["success"] == True else "Failed"])
        validation_results = ValidationResults.from_dict(validation_results_dict)
        return validation_results

    def filter_valid_constraints(self, code_list_for_constraints, spark,
                                 spark_df) -> list:
        check_result_on_original_validation_df = self.apply_checks_from_strings_on_spark_df(spark, spark_df,
                                                                                            code_list_for_constraints)
        status_on_original_validation_df = check_result_on_original_validation_df["results"]
        # remove the constraints that are not grammarly correct
        code_list_for_constraints = [code_list_for_constraints[i] for i in range(len(code_list_for_constraints)) if
                                     status_on_original_validation_df[i]["success"] == True]
        return code_list_for_constraints
