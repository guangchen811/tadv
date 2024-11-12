from pydeequ import Check, CheckLevel

from cadv_exploration.deequ import apply_checks_from_strings


def filter_constraints(code_list_for_constraints, spark_original_validation, spark_original_validation_df, logger):
    logger.info(f"Suggested Code list for constraints: {code_list_for_constraints}")
    check_result_on_original_validation_df = apply_checks_from_strings(code_list_for_constraints,
                                                                       spark_original_validation,
                                                                       spark_original_validation_df)
    status_on_original_validation_df = [item['constraint_status'] if
                                        item is not None else None for item in check_result_on_original_validation_df]
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
    logger.info(f"Grammarly failure check on original data: {grammarly_failure_check_output_on_original_validation_df}")
    # remove the constraints that are not grammarly correct
    code_list_for_constraints = [code_list_for_constraints[i] for i in range(len(code_list_for_constraints)) if
                                 status_on_original_validation_df[i] == "Success"]
    logger.info(f"Filtered Code list for constraints: {code_list_for_constraints}")
    return code_list_for_constraints


