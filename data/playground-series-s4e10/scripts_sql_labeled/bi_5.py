class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Segment customers based on income levels and loan grade
SELECT CASE
           WHEN person_income < 50000 THEN 'Low Income'
           WHEN person_income BETWEEN 50000 AND 100000 THEN 'Medium Income'
           ELSE 'High Income'
           END        AS income_segment,
       loan_grade,
       COUNT(*)       AS total_customers,
       AVG(loan_amnt) AS avg_loan_amount
FROM test
GROUP BY income_segment, loan_grade
ORDER BY income_segment, loan_grade;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['person_income', 'loan_grade', 'loan_amnt']
