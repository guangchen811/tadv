class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- One-hot encoding for categorical features (e.g., home ownership)
SELECT *,
       CASE WHEN person_home_ownership = 'RENT' THEN 1 ELSE 0 END     AS home_ownership_rent,
       CASE WHEN person_home_ownership = 'MORTGAGE' THEN 1 ELSE 0 END AS home_ownership_mortgage,
       CASE WHEN person_home_ownership = 'OWN' THEN 1 ELSE 0 END      AS home_ownership_own
FROM test;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade', 'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'person_age', 'person_emp_length', 'person_home_ownership',
                'person_income', 'loan_status']
