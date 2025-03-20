-- Create lag features for credit history length to capture trends
SELECT * EXCLUDE (person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_status, loan_percent_income, cb_person_default_on_file)
, cb_person_cred_hist_length - LAG(cb_person_cred_hist_length) OVER (ORDER BY id) AS credit_hist_change
FROM new_data;