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
    expectations = {}
    for idx, code in enumerate(code_list_for_constraints):
        try:
            expectation = eval(f"gx.expectations.{code}")
            suite.add_expectation(expectation)
            expectations[idx] = expectation
        except AttributeError as e:
            error_message = str(e)
            expectations[idx] = error_message
        except Exception as e:
            raise RuntimeError(
                f"Error while evaluating the expectation code '{code}': {str(e)}"
            )
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)
    validation_results = batch.validate(suite)
    # ordered_results = [
    #     result for expectation in expectations
    #     for result in validation_results.results
    #     if result.expectation_config.id == expectation.id
    # ]
    ordered_results = []
    for idx in range(len(code_list_for_constraints)):
        if isinstance(expectations[idx], gx.expectations.expectation.Expectation):
            for result in validation_results.results:
                if result.expectation_config.id == expectations[idx].id:
                    ordered_results.append(result)
                    break
        else:
            ordered_results.append(
                {
                    "success": False,
                    "expectation_config": {"type": "InvalidExpectation"},
                }
            )
    validation_results.results = ordered_results
    return validation_results
