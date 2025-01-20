class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
WITH admission_windows AS (
    SELECT
        "id",
        CAST("Date of Admission" AS DATE) AS admission_date,
        COUNT(*) OVER (
            PARTITION BY "id" 
            ORDER BY CAST("Date of Admission" AS DATE)
            RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
        ) AS count_30_day_window
    FROM train
)
SELECT
    "id",
    admission_date,
    count_30_day_window,
    CASE
        WHEN count_30_day_window > 1 THEN 1
        ELSE 0
    END AS frequent_visitor_flag
FROM admission_windows
ORDER BY "id", admission_date;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
