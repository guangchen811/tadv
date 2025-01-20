class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Create lag features for credit history length to capture trends
SELECT *,
       cb_person_cred_hist_length - LAG(cb_person_cred_hist_length) OVER (ORDER BY id) AS credit_hist_change
FROM test;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade', 'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'person_age', 'person_emp_length', 'person_home_ownership',
                'person_income', 'loan_status']
