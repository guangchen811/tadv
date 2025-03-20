-- Analyze loan performance and default rates by loan grade and loan intent
SELECT new_data.loan_grade,                                                                 -- Credit grade of the loan, indicating borrower risk level
       new_data.loan_intent,                                                                -- Purpose of the loan (e.g., education, medical, personal, home improvement)

       COUNT(*)                                                  AS total_loans,            -- Total number of loans issued for each grade and intent category
       SUM(CASE WHEN new_data.loan_status = 1 THEN 1 ELSE 0 END) AS total_defaults,         -- Total number of defaults in each category

       -- Calculate default rate as the percentage of loans that defaulted
       ROUND(SUM(CASE WHEN new_data.loan_status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                                                                 AS default_rate_percentage,

       ROUND(AVG(new_data.loan_amnt), 2)                         AS avg_loan_amount,        -- Average loan amount issued
       ROUND(AVG(new_data.person_income), 2)                     AS avg_borrower_income,    -- Average income of borrowers in each category
       ROUND(AVG(new_data.loan_percent_income), 2)               AS avg_loan_percent_income -- Average percentage of income allocated to loan repayment

FROM new_data
GROUP BY loan_grade,
         loan_intent -- Grouping by loan grade and loan intent to analyze trends across different risk levels and purposes

ORDER BY default_rate_percentage DESC, -- Prioritizing loan categories with higher default rates
         loan_grade,
         loan_intent;
-- Ensuring consistent order within each grade

/*
Other relevant columns in new_data:
- person_age: Could be used to analyze default trends by age group.
- person_home_ownership: Might provide insights into how home ownership affects default rates.
- cb_person_cred_hist_length: Longer credit history might correlate with lower default rates.
- person_emp_length: Job stability could impact a borrower's ability to repay loans.

These factors could be included in a deeper analysis of borrower risk profiles.
*/