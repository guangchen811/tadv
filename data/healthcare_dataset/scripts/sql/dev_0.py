class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "id",
    "Name",
    "Age",
    "Medical Condition",
    "Doctor",
    "Hospital"
FROM train
WHERE ("Name" ILIKE :search_term OR "Medical Condition" ILIKE :search_term)
ORDER BY "Name" ASC
LIMIT :limit_value
OFFSET :offset_value;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
