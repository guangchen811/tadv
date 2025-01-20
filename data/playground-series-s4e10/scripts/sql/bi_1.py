class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Analyze loan approval rates based on home ownership status
SELECT person_home_ownership,
       COUNT(*)                                                                          AS total_loans,
       SUM(CASE WHEN cb_person_default_on_file = 'N' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS approval_rate
FROM test
GROUP BY person_home_ownership
ORDER BY approval_rate DESC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['person_home_ownership', 'cb_person_default_on_file']
