class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Categorizing interest rates into tiers
SELECT *,
       CASE
           WHEN loan_int_rate < 10 THEN 'Low Rate'
           WHEN loan_int_rate BETWEEN 10 AND 20 THEN 'Medium Rate'
           ELSE 'High Rate'
       END AS interest_rate_tier
FROM train;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['id', 'cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade',
                'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'person_age', 'person_emp_length', 'person_home_ownership',
                'person_income', 'loan_status']
