class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "id",
    "Name",
    "Age",
    "Medical Condition",
    CASE
        WHEN "Age" >= 65 OR "Medical Condition" IN ('Cancer', 'Diabetes') THEN 'High'
        ELSE 'Low/Medium'
    END AS risk_level
FROM train
-- Return only the "High" risk patients
WHERE (
    "Age" >= 65
    OR "Medical Condition" IN ('Cancer', 'Diabetes')
)
ORDER BY "Age" DESC, "Medical Condition";
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
