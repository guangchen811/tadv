class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "Doctor",
    COUNT(*) AS patient_count,
    ROUND(AVG("Billing Amount"), 2) AS avg_billing
FROM train
WHERE (:hospital_filter IS NULL OR "Hospital" = :hospital_filter)
GROUP BY "Doctor"
ORDER BY patient_count DESC
LIMIT 50;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
