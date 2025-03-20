-- All columns in the new_data table are as follows:
-- id,person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt
-- loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length, loan_status

WITH employment_data AS (
    -- Select relevant columns from the new_data table
    SELECT p.person_emp_length AS employment_duration, -- Employment length of the borrower
           p.loan_grade        AS grade_segment,       -- Loan grade classification
           p.loan_status       AS status,              -- Loan status (1 for default, 0 for non-default)
           p.loan_int_rate     AS interest_rate        -- Interest rate for the loan
    FROM new_data p)

-- Aggregate and analyze loan data by employment duration and loan grade
SELECT employment_duration                         AS emp_length,        -- Employment duration of the borrower
       grade_segment,                                                    -- Loan grade classification
       COUNT(*)                                    AS total_loans,       -- Total number of loans in the segment
       ROUND(AVG(interest_rate), 2)                AS avg_interest_rate, -- Average interest rate in the segment
       SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS defaults,          -- Total number of defaults in the segment
       ROUND(SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
             2)                                    AS default_percentage -- Default rate as a percentage
FROM employment_data
GROUP BY employment_duration, grade_segment -- Group data by employment duration and loan grade
ORDER BY default_percentage DESC, emp_length, grade_segment;
-- Sort results by default percentage (highest first), then employment length, then grade
