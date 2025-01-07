class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Count loans by grade
SELECT loan_grade,
       COUNT(*) AS count_by_grade
FROM test
GROUP BY loan_grade
ORDER BY count_by_grade DESC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['loan_grade']
