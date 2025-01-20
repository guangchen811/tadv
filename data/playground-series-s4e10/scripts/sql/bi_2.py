class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
        -- Track default rates for different loan intents
SELECT loan_intent,
       COUNT(*)                                                                          AS total_loans,
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END)                  AS total_defaults,
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
FROM test
GROUP BY loan_intent
ORDER BY default_rate DESC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['loan_intent', 'cb_person_default_on_file']
