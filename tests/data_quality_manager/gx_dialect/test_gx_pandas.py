import great_expectations as gx
import pandas as pd


def test_gx_init():
    # Retrieve your Data Context
    context = gx.get_context()

    # Define the Data Source name
    data_source_name = "my_data_source"

    # Add the Data Source to the Data Context
    data_source = context.data_sources.add_pandas(name=data_source_name)

    # Define the Data Asset name
    data_asset_name = "my_dataframe_data_asset"

    # Add a Data Asset to the Data Source
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)

    # Define the Batch Definition name
    batch_definition_name = "my_batch_definition"

    # Add a Batch Definition to the Data Asset
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        batch_definition_name
    )

    dataframe = pd.DataFrame(
        {
            "passenger_count": [1, 2, 3, 4, 5],
            "trip_distance": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    # Define the Batch Parameters
    batch_parameters = {"dataframe": dataframe}

    suite_name = "my_expectation_suite"
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)
    expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="passenger_count", max_value=6, min_value=1
    )
    suite.add_expectation(expectation)
    expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="trip_distance", max_value=3, min_value=1
    )
    suite.add_expectation(expectation)

    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    validation_results = batch.validate(suite)
    print(validation_results)

    # validation_results_dict = {}
    # return ValidationResults.from_dict(validation_results_dict)
