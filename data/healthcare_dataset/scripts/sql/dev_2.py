class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "id",
    "Name",
    "Date of Admission",
    "Test Results",
    "Medical Condition",
    "Billing Amount"
FROM train
WHERE CAST("Date of Admission" AS DATE) BETWEEN CAST(:start_date AS DATE)
                                            AND CAST(:end_date AS DATE)
  AND "Test Results" = :test_result
ORDER BY "Date of Admission" ASC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
