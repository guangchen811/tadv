class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "Hospital",
    "Admission Type",
    ROUND(
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY "Hospital"),
        3
    ) AS admission_ratio
FROM train
GROUP BY "Hospital", "Admission Type"
ORDER BY "Hospital", admission_ratio DESC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
