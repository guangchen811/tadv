## End-to-End Data Error Impact

This experiment evaluates how effectively our system can detect data errors that impact downstream tasks.

### Experiment Setup

To conduct the experiment, we collect two types of results:

1. **Downstream Task Performance**
2. **Constraint Validation Results**

We generate these results by running the following scripts:

- [`calculate_code_performance.py`](./evaluation/calculate_code_performance.py) – gathers downstream performance
  results.
- [`validate_constraints.py`](./evaluation/validate_constraints.py) – validates the suggested constraints.

### Output Directories

- **Downstream Task Performance** results are stored in:  
  `data_processed/<dataset>/<downstream-task>/<processed-data-label>/output_validation/`

- **Constraint Validation** results are stored in:  
  `data_processed/<dataset>/<downstream-task>/<processed-data-label>/constraints_validation/`

We then use [`main.py`](./evaluation/main.py) to aggregate the results and save them to the [
`result_tables`](./result_tables) directory.

The results are visualized using the notebook:  
[`result_analysis.ipynb`](./virtualization/result_analysis.ipynb)

### Sample Result Table

| ID | Method | Variant         | TP  | FP  | FN | TN | F1 Score |
|----|--------|-----------------|-----|-----|----|----|----------|
| 0  | deequ  | None            | 95  | 370 | 41 | 4  | 0.337    |
| 1  | deequ  | column_skipped  | 328 | 137 | 34 | 11 | 0.816    |
| 2  | gpt-4o | None            | 373 | 92  | 22 | 23 | 0.866    |
| 3  | gpt-4o | with_deequ      | 344 | 121 | 33 | 12 | 0.838    |
| 4  | gpt-4o | with_experience | 374 | 91  | 28 | 17 | 0.874    |
| 5  | tfdv   | None            | 161 | 304 | 45 | 0  | 0.514    |