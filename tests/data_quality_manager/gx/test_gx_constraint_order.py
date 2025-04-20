import great_expectations as gx
import pandas as pd
import pyspark


def test_apply_checks_from_strings_on_spark_df():
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})
    spark = pyspark.sql.SparkSession.builder \
        .appName("Test") \
        .getOrCreate()
    spark_df = spark.createDataFrame(df)
    code_list_for_constraints = [
        'ExpectColumnValuesToNotBeNull(column="a")',
        'ExpectCompoundColumnsToBeUnique(column_list=["b", "c"])',
        'ExpectColumnValuesToBeInSet(column="a", value_set=["foo", "bar", "baz"])',
        'ExpectColumnToExist(column="def")',
        'ExpectColumnValuesToNotBeNull(column="c")',
    ]

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

    expectations = []
    for code in code_list_for_constraints:
        expectation = eval(f"gx.expectations.{code}")
        expectations.append(expectation)
        suite.add_expectation(expectation)

    batch = batch_definition.get_batch(batch_parameters=batch_parameters)
    validation_results = batch.validate(suite)
    ordered_results = [
        result for expectation in expectations
        for result in validation_results.results
        if result["expectation_config"].id == expectation.id
    ]
    for i, result in enumerate(ordered_results):
        print(f"{i + 1}. {code_list_for_constraints[i]} -> success: {result['success']}")
