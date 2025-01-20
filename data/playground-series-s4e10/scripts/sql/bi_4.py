class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Highlight loan intents with the highest approval volume
SELECT loan_intent,
       COUNT(*) AS approved_loans
FROM test
WHERE cb_person_default_on_file = 'N'
GROUP BY loan_intent
ORDER BY approved_loans DESC LIMIT 5;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['loan_intent', 'cb_person_default_on_file', 'loan_status']
