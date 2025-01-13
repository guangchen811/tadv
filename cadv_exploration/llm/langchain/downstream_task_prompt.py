from inspect import cleandoc

ML_INFERENCE_TASK_DESCRIPTION = cleandoc("""The code is written for ML Inference task. After training a model on the training data, the model would be used to make predictions on the test data.
you are asked to generate constraints on the upcoming *test* data to ensure that the code can run without any errors and the predictions are meaningful.""")

SQL_QUERY_TASK_DESCRIPTION = cleandoc(
    """The code is written for SQL Query task. The code is expected to run on a database and return the results.
you are asked to generate constraints on the upcoming *test* data to ensure that the code can run without any errors and the results are meaningful."""
)

CD_TASK_DESCRIPTION = cleandoc(
    """The code is designed to process the current version of the data and should also work seamlessly with future datasets. Your task is to write constraints that validate abnormalities in the data.

For categorical attributes, your goal is to provide a comprehensive and accurate list of all possible categories within the given domain.
	•	Expand the list to include all valid and expected categories, even if they are not present in the current version of the data. This ensures that the constraints avoid false alerts and account for valid categories in upcoming data.
	•	Ensure the list is complete, capturing all common and relevant categories.
	•	Use consistent formatting (e.g., lowercase) if specified.
	•	Avoid duplicates and irrelevant entries.
The categorical value here includes all the attributes that you can complete with common sense like colors, countries, states, seasons, etc.
Try your best to provide a comprehensive list of categories that can be used to validate the data.

Example:
If the domain involves colors and the input includes "red" and "blue", expand the output to include all possible categories:
"red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray"
    """)
