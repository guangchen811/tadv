class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Binning employment length into categories
SELECT *,
       CASE
           WHEN person_emp_length < 2 THEN 'Junior'
           WHEN person_emp_length BETWEEN 2 AND 5 THEN 'Mid-level'
           WHEN person_emp_length > 5 THEN 'Senior'
           ELSE 'Unknown'
       END AS emp_length_category
FROM test;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade', 'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'person_age', 'person_emp_length', 'person_home_ownership',
                'person_income', 'loan_status']