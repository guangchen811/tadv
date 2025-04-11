import great_expectations as gx


def apply_checks_from_strings_on_spark_df(spark, spark_df, code_list_for_constraints):
    context = gx.get_context()
    data_source_name = "my_data_source"
    data_asset_name = "my_dataframe_data_asset"
    batch_definition_name = "my_batch_definition"
    suite_name = "my_expectation_suite"
    data_source = context.data_sources.add_spark(name=data_source_name)
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        batch_definition_name
    )
    batch_parameters = {"dataframe": spark_df}
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)
    for code in code_list_for_constraints:
        exec(f"suite.add_expectation(gx.expectations.{code})")
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)
    validation_results = batch.validate(suite)
    return validation_results
