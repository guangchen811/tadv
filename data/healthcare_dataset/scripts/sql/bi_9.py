class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
WITH stays AS (
    SELECT
        CAST("Date of Admission" AS DATE) AS admission_date,
        CAST("Discharge Date" AS DATE) AS discharge_date,
        DATEDIFF('day', 
                 CAST("Date of Admission" AS DATE), 
                 CAST("Discharge Date" AS DATE)) AS length_of_stay,
        "Test Results"
    FROM train
)
SELECT
    "Test Results",
    ROUND(AVG(length_of_stay), 2) AS avg_stay_days,
    COUNT(*) AS patient_count
FROM stays
GROUP BY "Test Results"
ORDER BY avg_stay_days DESC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
