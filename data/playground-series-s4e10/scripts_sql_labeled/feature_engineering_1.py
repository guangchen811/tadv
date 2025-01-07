class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Adding a feature for the ratio of loan amount to annual income
SELECT loan_amnt * 1.0 / person_income AS income_to_loan_ratio
FROM test;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['loan_amnt', 'person_income']
