class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT * EXCLUDE (person_income)
FROM test
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade', 'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'person_age', 'person_emp_length', 'person_home_ownership',
                'loan_status']
