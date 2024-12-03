from inspect import cleandoc

ML_INFERENCE_TASK_DESCRIPTION = cleandoc("""The code is written for ML Inference task. After training a model on the training data, the model would be used to make predictions on the test data.
you are asked to generate constraints on the upcomming *test* data to ensure that the code can run without any errors and the predictions are meaningful.""")

SQL_QUERY_TASK_DESCRIPTION = cleandoc(
    """The code is written for SQL Query task. The code is expected to run on a database and return the results.
you are asked to generate constraints on the upcomming *test* data to ensure that the code can run without any errors and the results are meaningful."""
)
