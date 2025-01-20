class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
WITH doctor_metrics AS (
    SELECT
        "Doctor",
        COUNT(*) AS patient_count,
        ROUND(AVG("Billing Amount"), 2) AS avg_billing
    FROM train
    GROUP BY "Doctor"
)
SELECT
    "Doctor",
    patient_count,
    avg_billing,
    RANK() OVER (ORDER BY patient_count DESC) AS patient_count_rank,
    RANK() OVER (ORDER BY avg_billing DESC)   AS avg_billing_rank
FROM doctor_metrics
ORDER BY patient_count DESC, avg_billing DESC
LIMIT 10;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
