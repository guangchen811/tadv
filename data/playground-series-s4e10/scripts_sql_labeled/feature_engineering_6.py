class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Adding a loan-to-income ratio feature and binning it into categories
SELECT ROUND(loan_amnt * 1.0 / person_income, 2) AS loan_to_income_ratio,
       CASE
           WHEN loan_amnt * 1.0 / person_income < 0.1 THEN 'Low Ratio'
           WHEN loan_amnt * 1.0 / person_income BETWEEN 0.1 AND 0.3 THEN 'Moderate Ratio'
           ELSE 'High Ratio'
       END AS loan_to_income_band
FROM train;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['loan_amnt', 'person_income']
