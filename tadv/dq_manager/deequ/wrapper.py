from tadv.data_models import Constraints, ValidationResults
from tadv.dq_manager.abstract_data_quality_manager import ConstraintSuggestingDataQualityManager
from tadv.dq_manager.deequ._analyzing import analyze_on_spark_df
from tadv.dq_manager.deequ._constraint_suggestion import \
    get_suggestion_for_spark_df
from tadv.dq_manager.deequ._constraint_validation import apply_checks_from_strings_on_spark_df
from tadv.dq_manager.deequ._profiling import profile_on_spark_df


class DeequDataQualityManager(ConstraintSuggestingDataQualityManager):
    def __init__(self):
        super().__init__()

    @staticmethod
    def analyze_on_spark_df(spark, spark_df, analyzers):
        return analyze_on_spark_df(spark, spark_df, analyzers)

    @staticmethod
    def profile_on_spark_df(spark, spark_df):
        """
        This function is based on the profiling function from Deequ. So it couldn't be implemented by great expectations.
        """
        return profile_on_spark_df(spark, spark_df)

    @staticmethod
    def apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints, isolated_check=True):
        return apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints, isolated_check)

    def validate_on_spark_df(self, spark, spark_df, code_list_for_constraints, return_raw=False, isolated_check=True):
        check_result = apply_checks_from_strings_on_spark_df(spark, spark_df,
                                                             code_list_for_constraints=code_list_for_constraints,
                                                             isolated_check=isolated_check)
        if return_raw:
            return check_result
        status = [item['constraint_status'] if
                  item is not None else None for item in check_result]
        return status

    def inference_constraints_for_spark_df(self, spark, spark_df, spark_validation=None,
                                           spark_validation_df=None) -> Constraints:
        """
        This function is based on the suggestion from Deequ. So it couldn't be implemented by great expectations.
        """
        suggestion = self._get_suggestion_for_spark_df(spark, spark_df)
        code_list_for_constraints = [item["code_for_constraint"] for item in suggestion]
        if spark_validation is None or spark_validation_df is None:
            code_list_for_constraints_valid = self.filter_valid_constraints_on_spark(code_list_for_constraints, spark,
                                                                                     spark_df)
        else:
            code_list_for_constraints_valid = self.filter_valid_constraints_on_spark(code_list_for_constraints,
                                                                                     spark_validation,
                                                                                     spark_validation_df)
        constraints = Constraints.from_deequ_output(suggestion, code_list_for_constraints_valid)
        return constraints

    def filter_valid_constraints_on_spark(self, code_list_for_constraints, spark,
                                          spark_df) -> list:
        check_result_on_original_validation_df = self.apply_checks_from_strings_on_spark_df(spark, spark_df,
                                                                                            code_list_for_constraints)
        status_on_original_validation_df = [item['constraint_status'] if
                                            item is not None else None for item in
                                            check_result_on_original_validation_df]
        # remove the constraints that are not grammarly correct
        code_list_for_constraints = [code_list_for_constraints[i] for i in range(len(code_list_for_constraints)) if
                                     status_on_original_validation_df[i] == "Success"]
        return code_list_for_constraints

    @staticmethod
    def build_validation_results(code_list_for_constraints, status, valid_code_column_map):
        code_status_map = {code_list_for_constraints[i]: status[i] for i in
                           range(len(code_list_for_constraints))}
        validation_results_dict = {"results": {column: {"code": []} for column in valid_code_column_map.values()}}
        for code, column in valid_code_column_map.items():
            validation_results_dict["results"][column]["code"].append(
                [code, "Passed" if code_status_map[code] == "Success" else "Failed"])
        validation_results = ValidationResults.from_dict(validation_results_dict)
        return validation_results

    @staticmethod
    def _get_suggestion_for_spark_df(spark, spark_df):
        return get_suggestion_for_spark_df(spark, spark_df)
