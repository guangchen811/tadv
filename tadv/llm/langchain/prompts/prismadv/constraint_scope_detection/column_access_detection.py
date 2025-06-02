from inspect import cleandoc

COLUMN_ACCESS_DETECTION_PROMPT = cleandoc("""You are part of the task-aware data validation system. You serve as the *Column Access Detection* component.
Given a dataset and the downstream code, you are asked to find the columns that are used in the code snippet. These columns are the accessed columns for the downstream task to ensure that the constraints are only applied to accessed columns.

The dataset is a table with the following columns:
{columns_desc}

The user writes the code snippet below:
{code_snippet}

The above code snippet is used for the following downstream task:
{downstream_task_description}

Your response should be a list of comma separated values
eg: `foo, bar, baz` or `foo,bar,baz`
""")
