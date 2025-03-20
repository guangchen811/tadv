# Task-aware Data Validation (TADV)

[![codecov](https://codecov.io/github/guangchen811/tadv/graph/badge.svg?token=UC6B33P10M)](https://codecov.io/github/guangchen811/tadv)

## Overview

TADV is a framework for evaluating the data validation capabilities of large language models (LLMs) using contextual
information, such as downstream queries and machine learning pipelines.

### Project Structure

The project consists of the following [modules](/tadv):

- **[error_injection](/tadv/error_injection)** – Provides APIs for injecting errors into datasets, enabling robustness
  testing for validation methods.
- **[runtime_environments](/tadv/runtime_environments)** – Defines execution environments where datasets are evaluated
  in the context of downstream queries or machine learning pipelines.
- **[llm](/tadv/llm)** – Contains classes for interacting with LLM APIs to generate data validation rules. This
  process follows three key steps:
    1. **Target Column Detection** – Identifying relevant columns based on downstream context.
    2. **Assumption Generation** – Inferring data assumptions from provided context and dataset properties.
    3. **Rule Generation** – Producing executable validation rules to ensure data quality.
- **[inspector](/tadv/inspector)** – Extracts dataset metadata, including schema and statistics, to aid LLMs in
  generating informed validation rules.

#### Error Injection

The error injection module is built based on [Jenga](https://github.com/schelterlabs/jenga), a library for injecting
errors into datasets. We extend the error injection methods into more real world scenarios where we often need
context information to fix the errors. You can find the error injection
methods [here](/tadv/error_injection/corrupts).

The following table lists the error injection methods we support:

| **Type**                                                                                 | **Explanation**                                                                                               |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [Missing categorical value](/tadv/error_injection/corrupts/categorical_value_missing.py) | Replace one or more types of categorical value with a missing value or other existing values, or delete them. |
| [Dropping column](/tadv/error_injection/corrupts/column_dropping.py)                     | Drop one or more columns.                                                                                     |
| [Inserting column](/tadv/error_injection/corrupts/column_inserting.py)                   | Insert one or more columns by copying existing columns or generating new columns.                             |
| [Adding gaussian noise](/tadv/error_injection/corrupts/gaussian_noise.py)                | Add Gaussian noise to numerical values.                                                                       |
| [Masking values](/tadv/error_injection/corrupts/masking_values.py)                       | Mask one or more values in the dataset.                                                                       |
| [Scaling values](/tadv/error_injection/corrupts/scaling_values.py)                       | Scale numerical values by a factor.                                                                           |

#### Runtime Environments

The runtime environments module provides execution environments for evaluating datasets in the context of downstream.
TODO: add more details.

#### LLM

Currently, we use [langchain](https://www.langchain.com/) as the tool for llm api calls. We plan to extend it
to [dspy](https://dspy-docs.vercel.app/) in the future.

As shown in the following figure, we decompose the data validation task into two three sub-tasks:

- Target column detection: detect the target column that needs to be validated based on the downstream queries or
  machine learning pipelines.
- Assumption generation: generate assumptions based on the target column and the context information.
- Rule generation: generate formal rules in the form
  of [deequ](https://github.com/awslabs/python-deequ/blob/master/pydeequ/checks.py) for evaluation.

The prompts during the API calls can be found [here](/tadv/llm/langchain/_prompt.py). For more details, you
can look at the [test case](/tests/llm/langchain).

TODO: add more details.

#### Inspector

The inspector module is designed to provide dataset information, such as schema and statistics. It is built based
on [pydeequ](https://github.com/awslabs/python-deequ) and [pandas](https://pandas.pydata.org/). TODO: add more details.

## Provided Datasets

We provide two tabular datasets from Kaggle for testing the framework:

- [Healthcare](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- [Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10)

Besides, we also have a [toy dataset](/data/toy_example) for showcasing the whole workflow.

These datasets are available in the [data](/data) directory. In addition to the raw data, we provide scripts that run in
the [runtime environments](/tadv/runtime_environments) to evaluate the generated data validation rules.

### Example: [Healthcare Dataset](/data/healthcare_dataset)

The dataset is structured as follows:

- **`files/`** – Contains the source data.
- **`scripts/`** – Includes downstream scripts spanning three domains:
    - SQL queries
    - Machine learning pipelines
    - Website generation
- **`errors/`** – Stores error configurations used for error injection.
- **`annotations/`** – Provides dataset annotations, including:
    - Target columns for all scripts in the three domains
    - Assumptions associated with the target columns

## Experiment Workflow

We provide the following workflow for evaluating the data validation capabilities of LLMs compared to non-LLM methods.
You can find the detailed implementation in the [workflow](workflow) directory.

### Step 0: Environment Setup

### Install the package

We use [poetry](https://python-poetry.org/) to manage the dependencies. If you are not familiar with poetry, we suggest
you install it with [pipx](https://pipx.pypa.io/stable/) first by following
the [official documentation](https://python-poetry.org/docs/).

After installing poetry, you can install the dependencies by running the following command:

```shell
poetry install
```

You can then test the installation by running the following command. It will run all the [tests](/tests) in the project.

```shell
poetry run pytest
```

### Step 1: Preprocessing

To prepare the dataset for data validation, we need to preprocess the data in two steps:

- **Error Injection**: Inject errors into the dataset to simulate real-world data quality issues.
- **Script Execution**: Execute the downstream scripts to generate the ground truth for data validation.

#### 1.1 Remove Existing Preprocessed Data

We provide all the preprocessed data in the `data_processed/` folder for paper reviewing. If you want to reproduce the
results, you need to delete the existing preprocessed data first by running the following command:

```shell
rm -rf data_processed/*
```

#### 1.2 Errors Injection

To inject errors into the dataset, run the following command:

```shell
poetry run python ./workflow/s1_preprocessing/error_injection/main.py --dataset-option "all" --downstream-task-option "all"
```

This command will inject errors into the dataset in `data/` folder and then save the corrupted dataset in
`data_processed/` folder. The predefined error injection configurations can be found in `data/<dataset>/errors/`. You
could also customize the error injection configurations by modifying/adding the error injection scripts in the same
folder. Please make sure the name of the error injection script is started with `<downstream-task>_`, e.g.,
`ml_inference_classification_1.yaml`.

#### 1.3 Scripts Execution

To Execute the downstream scripts, run the following command:

```shell
poetry run python ./workflow/s1_preprocessing/scripts_execution/main.py --dataset-option "all" --downstream-task-option "all" --processed-data-label "0"
```

This command will execute the downstream scripts in `data/<dataset>/scripts/` and then save the results in the
`data_processed/<dataset>/<downstream-task>/<processed-data-label>/` folder.

### Step 2: Data Validation Rule Generation

#### 2.1 Target Column Detection

To detect the target column, run the following command:

```shell
poetry run python ./workflow/s2_experiments/t1_target_column_detection/run_langchain_tcd.py --dataset-option "all" --downstream-task-option "all" --processed-data-label "0"
```

#### 2.2 End-to-End Data Validation Rule Generation

To generate data validation rules, run the following command:

```shell
poetry run python ./workflow/s2_experiments/t2_constraint_inference/run_deequ_dv.py --dataset-option "all" --downstream-task-option "all" --processed-data-label "0"
```

```shell
poetry run python ./workflow/s2_experiments/t2_constraint_inference/run_langchain_tadv.py --dataset-option "all" --downstream-task-option "all" --processed-data-label "0"
```

### Step 3: Evaluation

#### 3.1 scripts Performance Evaluation

To evaluate the performance of the scripts in the downstream tasks, run the following command:

```shell
poetry run python ./workflow/s3_evaluation/evaluation/calculate_code_performance.py --dataset-option "all" --downstream-task-option "all" --processed-data-label "0"
```

The evaluation results will be saved in the `data_processed/<dataset>/<downstream-task>/<processed-data-label>/output_validation/` folder.

```shell
poetry run python ./workflow/s3_evaluation/evaluation/validate_constraints.py --dataset-option "all" --downstream-task-option "all" --processed-data-label "0"
```

The evaluation results will be saved in the `data_processed/<dataset>/<downstream-task>/<processed-data-label>/constraints_validation/` folder.

Now, you can aggregate the evaluation results by running the following command:

```shell
poetry run python ./workflow/s3_evaluation/evaluation/main.py --dataset-option "all" --downstream-task-option "all" --processed-data-label "0"
```
