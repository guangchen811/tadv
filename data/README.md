# Dataset

We provide two tabular datasets from Kaggle for testing the framework:

- [Healthcare](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- [Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10)

Besides, we also have a [toy dataset](/data/toy_example) for showcasing the whole workflow.

* The Loan Approval Prediction dataset is from the Kaggle competition "Playground Series - S4E10". Due to the Kaggle
  competition rules, we cannot provide the dataset directly in this repository. However, you can download it from
  this [link](https://www.kaggle.com/competitions/playground-series-s4e10) to download it.

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
    - Accessed columns for all scripts in the three domains
    - Assumptions associated with the accessed columns
