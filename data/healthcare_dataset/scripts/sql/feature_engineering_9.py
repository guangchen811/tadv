class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "id",
    "Medical Condition",
    "Admission Type",
    "Billing Amount",
    DATEDIFF(
        'day',
        CAST("Date of Admission" AS DATE),
        CAST("Discharge Date"    AS DATE)
    ) AS length_of_stay,
    CASE
        WHEN DATEDIFF(
            'day',
            CAST("Date of Admission" AS DATE),
            CAST("Discharge Date"    AS DATE)
        ) = 0 THEN NULL
        ELSE ROUND(
            "Billing Amount" 
            / DATEDIFF(
                'day',
                CAST("Date of Admission" AS DATE),
                CAST("Discharge Date"    AS DATE)
            ),
            2
        )
    END AS cost_per_day
FROM train
ORDER BY "id";
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
