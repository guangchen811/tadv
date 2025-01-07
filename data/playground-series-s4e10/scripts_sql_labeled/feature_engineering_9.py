class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Encoding loan grades as numeric values for modeling
SELECT *,
       CASE
           WHEN loan_grade = 'A' THEN 1
           WHEN loan_grade = 'B' THEN 2
           WHEN loan_grade = 'C' THEN 3
           WHEN loan_grade = 'D' THEN 4
           WHEN loan_grade = 'E' THEN 5
           WHEN loan_grade = 'F' THEN 6
           WHEN loan_grade = 'G' THEN 7
           ELSE NULL
           END AS grade_numeric
FROM test;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['id', 'cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade',
                'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'person_age', 'person_emp_length', 'person_home_ownership',
                'person_income', 'loan_status']
