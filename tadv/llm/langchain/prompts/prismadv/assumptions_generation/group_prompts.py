from inspect import cleandoc

COMPLETENESS_PROMPT = cleandoc("""Completeness is a measure of whether the data is complete and contains all the necessary information.
We Provide the following functions to generate the completeness constraints:
{short_description_completeness_functions}""")

SINGLE_COLUMN_NUMERIC_PROMPT = cleandoc("""Single column numeric constraints are used to check the validity of a single column in the dataset. We provide the following functions to generate the single column numeric constraints:
{short_description_single_column_numeric_functions}""")

MULTI_COLUMN_NUMERIC_PROMPT = cleandoc("""Multi column numeric constraints are used to check the validity of multiple columns in the dataset. We provide the following functions to generate the multi column numeric constraints:
{short_description_multi_column_numeric_functions}""")

SCHEMA_CONSTRAINT_PROMPT = cleandoc("""Schema constraints are used to check the validity of the schema of the dataset. We provide the following functions to generate the schema constraints:
{short_description_schema_functions}""")

DISTINCTNESS_PROMPT = cleandoc("""Distinctness constraints are used to check the validity of the distinctness of the dataset. We provide the following functions to generate the distinctness constraints:
{short_description_distinctness_functions}""")

UNIQUENESS_PROMPT = cleandoc("""Uniqueness constraints are used to check the validity of the uniqueness of the dataset. We provide the following functions to generate the uniqueness constraints:
{short_description_uniqueness_functions}""")

SINGLE_COLUMN_TEXT_PROMPT = cleandoc("""Single column text constraints are used to check the validity of a string column in the dataset. We provide the following functions to generate the single column text constraints:
{short_description_single_column_text_functions}""")

MULTI_COLUMN_TEXT_PROMPT = cleandoc("""Multi column text constraints are used to check the validity of multiple string columns in the dataset. We provide the following functions to generate the multi column text constraints:
{short_description_multi_column_text_functions}""")

SINGLE_COLUMN_CATEGORICAL_PROMPT = cleandoc("""Single column categorical constraints are used to check the validity of a categorical column in the dataset. We provide the following functions to generate the single column categorical constraints:
{short_description_single_column_categorical_functions}""")

MULTI_COLUMN_CATEGORICAL_PROMPT = cleandoc("""Multi column categorical constraints are used to check the validity of multiple categorical columns in the dataset. We provide the following functions to generate the multi column categorical constraints:
{short_description_multi_column_categorical_functions}""")

VOLUME_PROMPT = cleandoc("""Volume constraints are used to check the validity of the volume of the dataset. We provide the following functions to generate the volume constraints:
{short_description_volume_functions}""")
