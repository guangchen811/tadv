class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Income band analysis
SELECT CASE
           WHEN person_income < 50000 THEN 'Low Income'
           WHEN person_income BETWEEN 50000 AND 100000 THEN 'Medium Income'
           ELSE 'High Income'
           END            AS income_band,
       COUNT(*)           AS total_loans,
       AVG(loan_amnt)     AS avg_loan_amount,
       AVG(loan_int_rate) AS avg_interest_rate
FROM test
GROUP BY income_band
ORDER BY total_loans DESC;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['person_income', 'loan_amnt', 'loan_int_rate']
