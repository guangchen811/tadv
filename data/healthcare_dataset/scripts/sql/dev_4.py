class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "Hospital",
    "Medical Condition",
    ROUND(AVG("Billing Amount"), 2) AS avg_billing
FROM train
GROUP BY GROUPING SETS (
    ("Hospital", "Medical Condition"),
    ("Hospital"),
    ("Medical Condition")
)
ORDER BY "Hospital" NULLS FIRST, "Medical Condition" NULLS FIRST;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
