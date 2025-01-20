class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Adding a derived feature for default probability based on intent and income
SELECT *,
       CASE loan_intent
           WHEN 'DEBTCONSOLIDATION' THEN 0.25
           WHEN 'EDUCATION' THEN 0.15
           WHEN 'VENTURE' THEN 0.35
           ELSE 0.20
           END +
       CASE
           WHEN person_income < 50000 THEN 0.1
           WHEN person_income BETWEEN 50000 AND 100000 THEN 0.05
           ELSE -0.05
           END AS default_probability_score
FROM train;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['id', 'cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade',
                'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'person_age', 'person_emp_length', 'person_home_ownership',
                'person_income', 'loan_status']
