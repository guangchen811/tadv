# Task-aware Data Validation (TADV)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python](https://img.shields.io/badge/python-3.12-blue)
[![CI](https://github.com/guangchen811/tadv/actions/workflows/ci.yml/badge.svg?branch=main)](.github/workflows/ci.yml)
[![codecov](https://codecov.io/github/guangchen811/tadv/graph/badge.svg?token=UC6B33P10M)](https://codecov.io/github/guangchen811/tadv)

## Overview

TADV is a framework for evaluating the data validation capabilities of large language models using contextual
information, such as downstream queries and machine learning pipelines.

## Experiment Reproduction

Here we provide the codebase for reproducing the experiments in the paper.

| Section                       | Source Code                                                                                                       |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------|
| <div align="center">4.1</div> | <div align="center">[Column Access Detection](workflow/s2_experiments/t1_column_access_detection)</div>           |
| <div align="center">4.2</div> | <div align="center">[End-to-End Data Error Impact](workflow/s3_evaluation)</div>                                  |
| <div align="center">4.3</div> | <div align="center">[Uncovering Implicit Data Assumptions](workflow/s2_experiments/t2_constraint_inference)</div> |

## Project Structure

The project consists of the following [modules](tadv):

- **[Error Injection](tadv/error_injection)** – Provides APIs for injecting errors into datasets, enabling robustness
  testing for validation methods.
- **[Runtime Environments](tadv/runtime_environments)** – Defines execution environments where datasets are evaluated
  in the context of downstream queries or machine learning pipelines.
- **[LLM](tadv/llm)** – Contains classes for interacting with LLM APIs to generate data validation rules. This
  process follows three key steps:
    1. **Column Access Detection** – Identifying relevant columns based on downstream context.
    2. **Assumption Generation** – Inferring data assumptions from provided context and dataset properties.
    3. **Rule Generation** – Producing executable validation rules to ensure data quality.
- **[Inspector](tadv/inspector)** – Extracts dataset metadata, including schema and statistics, to aid LLMs in
  generating informed validation rules.

## Experiment Workflow

We provide the following workflow for evaluating the data validation capabilities of LLMs compared to non-LLM methods.
You can find the detailed implementation in the [workflow](workflow) directory.

### Step 0: Environment Setup

### Create a `.env` file

To run the experiments, you need to create a `.env` file in the root directory of the project. The `.env` file should
contain the following environment variables:

```env
HF_TOKEN=***
OPENAI_API_KEY=***
SPARK_VERSION=3.5
```

Please replace `***` with your own API keys.

### Install the package

We use [poetry](https://python-poetry.org/) to manage the dependencies. If you are not familiar with poetry, we suggest
you install it with [pipx](https://pipx.pypa.io/stable/) first by following
the [official documentation](https://python-poetry.org/docs/).

After installing poetry, you can install the dependencies by running the following command:

```shell
poetry install --with test
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
rm -r data_processed/*
```

#### 1.2 Errors Injection

To inject errors into the dataset, run the following command:

```shell
poetry run python ./workflow/s1_preprocessing/error_injection/main.py \
  --dataset-option "all" \
  --downstream-task-option "all"
```

This command will inject errors into the dataset in `data/` folder and then save the corrupted dataset in
`data_processed/` folder. The predefined error injection configurations can be found in `data/<dataset>/errors/`. You
could also customize the error injection configurations by modifying/adding the error injection scripts in the same
folder. Please make sure the name of the error injection script is started with `<downstream-task>_`, e.g.,
`ml_inference_classification_1.yaml`.

#### 1.3 Scripts Execution

To Execute the downstream scripts, run the following command:

```shell
poetry run python ./workflow/s1_preprocessing/scripts_execution/main.py \
  --dataset-option "all" \
  --downstream-task-option "all" \
  --processed-data-label "0"
```

This command will execute the downstream scripts in `data/<dataset>/scripts/` and then save the results in the
`data_processed/<dataset>/<downstream-task>/<processed-data-label>/` folder.

### Step 2: Data Validation Rule Generation

#### 2.1 Column Access Detection

To detect the accessed column, run the following command:

```shell
poetry run python ./workflow/s2_experiments/t1_accessed_column_detection/run_langchain_tcd.py \
  --dataset-option "all" \
  --downstream-task-option "all" \
  --processed-data-label "0"
```

#### 2.2 End-to-End Data Validation Rule Generation

To generate data validation rules, run the following command:

```shell
poetry run python ./workflow/s2_experiments/t2_constraint_inference/run_deequ_dv.py \
  --dataset-option "all" \
  --downstream-task-option "all" \
  --processed-data-label "0"
```

```shell
poetry run python ./workflow/s2_experiments/t2_constraint_inference/run_langchain_tadv.py \
  --dataset-option "all" \
  --downstream-task-option "all" \
  --processed-data-label "0"
```

### Step 3: Evaluation

#### 3.1 scripts Performance Evaluation

To evaluate the performance of the scripts in the downstream tasks, run the following command:

```shell
poetry run python ./workflow/s3_evaluation/evaluation/calculate_code_performance.py \
  --dataset-option "all" \
  --downstream-task-option "all" \
  --processed-data-label "0"
```

The evaluation results will be saved in the
`data_processed/<dataset>/<downstream-task>/<processed-data-label>/output_validation/` folder.

```shell
poetry run python ./workflow/s3_evaluation/evaluation/validate_constraints.py \
  --dataset-option "all" \
  --downstream-task-option "all" \
  --processed-data-label "0"
```

The evaluation results will be saved in the
`data_processed/<dataset>/<downstream-task>/<processed-data-label>/constraints_validation/` folder.

Now, you can aggregate the evaluation results by running the following command:

```shell
poetry run python ./workflow/s3_evaluation/evaluation/main.py \
  --dataset-option "all" \
  --downstream-task-option "all" \
  --processed-data-label "0"
```
