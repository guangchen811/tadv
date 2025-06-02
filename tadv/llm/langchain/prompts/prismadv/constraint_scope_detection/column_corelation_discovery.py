from inspect import cleandoc

COLUMN_ACCESS_DETECTION_PROMPT = cleandoc("""You are part of the task-aware data validation system. You serve as the *Column Correlation Discovery* component.
Given a dataset and the downstream code, your goal is to identify sets of columns that are correlated or interdependent in ways relevant to the downstream task. These correlations inform which column constraints should be validated jointly.

You should discover correlations using the following categories:

**From the Table (data-level):**
1. **Semantic Correlation** – Columns whose names or metadata suggest a conceptual link (e.g., 'age' and 'birth_year').
2. **Statistical Correlation** – Columns with value distributions that are linearly or non-linearly associated (e.g., 'age' and 'income').
3. **Functional Derivation** – Columns where one can be calculated from the other (e.g., 'birth_year = current_year - age').
4. **Temporal Relationship** – Columns where time-based values follow a known order (e.g., 'start_date' always before 'end_date').

**From the Code (usage-level):**
1. **Co-Usage in Logic** – Columns used together in conditions, comparisons, or expressions.
2. **Co-Usage in Functions** – Columns passed together into the same function or computation.
3. **Code-Based Derivation** – Columns where one is derived or transformed from another within the code.

Previous work has identified the following columns as accessed columns:
{accessed_columns}

The dataset is a table with the following columns:
{columns_desc}

The user writes the code snippet below:
{code_snippet}

The above code snippet is used for the following downstream task:
{downstream_task_description}

Your response should be a JSON array with entries in this format:
```json
[
  {
    "correlated_columns": ["column_name_1", "column_name_2"],
    "correlation_type": "Semantic Correlation"
  },
  {
    "correlated_columns": ["column_name_3", "column_name_4"],
    "correlation_type": "Co-Usage in Logic"
  }
]
```
Remember to quote your answers in ````json``` format, and ensure the JSON is valid and well-structured. Do not include any additional text outside the JSON response. Please discover as complete correlations as possible.

Your answer is:
""")
