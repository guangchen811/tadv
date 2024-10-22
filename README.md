[![codecov](https://codecov.io/gh/guangchen811/cadv-exploration/graph/badge.svg?token=UC6B33P10M)](https://codecov.io/gh/guangchen811/cadv-exploration)

# Context-aware Data Validation

## Overview

This repository is designed for exploring the data validation ability of llms with context information (e.g., downstream
queries, downstream ml pipelines, etc.).

This repository contains the following components:

- [error_injection](./cadv_exploration/error_injection): contains the APIs for injecting errors into datasets.
- [runtime_environments](./cadv_exploration/runtime_environments): contains the API class for runtime environments,
  where datasets can be evaluated on downstream queries or machine learning pipelines.
- [inspector](./cadv_exploration/inspector): designed to provide dataset information, such as schema and statistics, to
  help LLMs generate data validation rules.
- [llms](./cadv_exploration/llm): contains the prompts and classes for making API calls to LLMs.

## error injection

<img src="./assets/error_injection.png" alt="error_injection" width="30%"/>

The error injection module is built based on [Jenga](https://github.com/schelterlabs/jenga), a library for injecting
errors into datasets. We plan to extend the error injection methods into more real world scenarios where we often need
context information to fix the errors.

The errors are shown in the following table:

| **Type**                      | **Explanation**                                                                    |
|-------------------------------|------------------------------------------------------------------------------------|
| Data formatting               | Change the format of data in the original dataset (e.g., DD/MM/YYYY vs MM/DD/YYYY) |
| Missing categorical value     | Delete some values when recognizing a categorical column.                          |
| Violated attribute dependency | Create columns that conflict with attribute dependency.                            |

## runtime environments

We use kaggle as the first runtime environment to evaluate the generated data validation rules. The kaggle runtime
environment is downloaded from their [official github repository](https://github.com/Kaggle/docker-python). As shown in
the following figure, we run the user ipynb codes on the kaggle dataset with kaggle runtime environment. The output
contains two parts, a new notebook with the outputs and the submission csv file. You can look
at [this test case](./tests/runtime/kaggle/test_runnable.py) for more details.

<img src="./assets/runtime_environments.png" alt="runtime_environments" width="50%"/>

## inspector

The inspector module is designed to provide dataset information, such as schema and statistics. It is built based
on [pydeequ](https://github.com/awslabs/python-deequ) and [pandas](https://pandas.pydata.org/).

## llms

Currently, we use [langchain](https://www.langchain.com/) as the tool for llm api calls. We plan to extend it
to [dspy](https://dspy-docs.vercel.app/) in the future.

As shown in the following figure, we decompose the data validation task into two three sub-tasks:

- Target column detection: detect the target column that needs to be validated based on the downstream queries or
  machine learning pipelines.
- Assumption generation: generate assumptions based on the target column and the context information.
- Rule generation: generate formal rules in the form
  of [deequ](https://github.com/awslabs/python-deequ/blob/master/pydeequ/checks.py) for evaluation.

<img src="./assets/llm_framework.png" alt="llm_framework" width="50%"/>

The prompts during the API calls can be found [here](./cadv_exploration/llm/langchain/_prompt.py).

# Other Thoughts

If we can use provenance techniques to build a dataset that labels which columns are used by which queries or codes. We
can use it to train a model that can predict which columns are likely to be used by a new query or code.

How to leverage SchemaPile to train a model to predict which columns are likely to be used to add checks? checks can be
treated as a node-level classification problem. We can use the schema information to build a graph where each node is a
column and each edge is (a foreign key relationship or other relationships). We can use the graph to train a model that
can predict which columns are likely to be used by a new query or code.

Which relations can be used to connect columns into a graph? may be some function dependencies, correlations, semantic
relationships. What are the node features? column name, data type, and other metadata. What are the edge features?
foreign key relationships, other relationships.

Raha paper gives me some references about how to classify errors in data. It is a good start point. I can extend it when
context is given to make the task more formal.

I should also see how context can be helpful for the error type that already mentioned in raha.