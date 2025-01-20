class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Combine multiple risk factors to identify high-risk segments
SELECT person_home_ownership,
       loan_grade,
       COUNT(*)                                                                          AS total_loans,
       AVG(loan_int_rate)                                                                AS avg_interest_rate,
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
FROM test
WHERE loan_int_rate > 15.0
GROUP BY person_home_ownership, loan_grade
HAVING default_rate > 0.2
ORDER BY default_rate DESC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['person_home_ownership', 'loan_grade', 'loan_int_rate', 'cb_person_default_on_file']
