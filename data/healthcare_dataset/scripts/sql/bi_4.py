class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "Insurance Provider",
    "Medical Condition",
    ROUND(AVG("Billing Amount"), 2) AS avg_billing
FROM train
GROUP BY GROUPING SETS (
    ("Insurance Provider", "Medical Condition"),
    ("Insurance Provider"),
    ("Medical Condition")
)
ORDER BY "Insurance Provider", "Medical Condition";
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
