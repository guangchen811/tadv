class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    t."id",
    SUM(CASE WHEN t."Medical Condition" = 'Cancer' THEN 3 ELSE 0 END
        + CASE WHEN t."Medical Condition" = 'Diabetes' THEN 2 ELSE 0 END
        + CASE WHEN t."Medical Condition" = 'Hypertension' THEN 1 ELSE 0 END
    ) AS condition_risk_score
FROM train AS t
GROUP BY t."id"
ORDER BY t."id";
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
