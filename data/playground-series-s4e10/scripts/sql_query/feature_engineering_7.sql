-- Categorizing interest rates into tiers
SELECT * EXCLUDE (person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_status, loan_percent_income, cb_person_default_on_file) CASE
           WHEN loan_int_rate < 10 THEN 'Low Rate'
           WHEN loan_int_rate BETWEEN 10 AND 20 THEN 'Medium Rate'
           ELSE 'High Rate'
END
AS interest_rate_tier
FROM new_data;