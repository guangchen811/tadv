/*
We already did an analysis on person_age and loan_grade. This query focuses on loan intents and their approval volume.
*/
-- Analyzing loan data with slightly obscured column references
WITH loan_data AS (SELECT l.loan_grade          AS grade_col,    -- Loan grade, representing borrower credit risk
                          l.loan_intent         AS intent_col,   -- Purpose of the loan (e.g., education, medical, personal, home improvement)
                          l.loan_status         AS status_col,   -- Loan status, used to identify defaults
                          l.loan_int_rate       AS int_rate_col, -- Interest rate applied to the loan
                          l.loan_amnt           AS amount_col,   -- Loan amount issued to the borrower
                          l.person_income       AS income_col,   -- Borrower's income level
                          l.loan_percent_income AS percent_col   -- Loan amount as a percentage of borrower's income
                   FROM new_data l)

SELECT grade_col                                       AS grade_segment,     -- Display segment by loan grade
       intent_col                                      AS intent_segment,    -- Display segment by loan intent

       COUNT(*)                                        AS total_loans,       -- Total number of loans issued
       SUM(CASE WHEN status_col = 1 THEN 1 ELSE 0 END) AS total_defaults,    -- Count of defaulted loans

       -- Calculate default percentage
       ROUND(SUM(CASE WHEN status_col = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                                                       AS default_rate,

       ROUND(AVG(int_rate_col), 2)                     AS avg_interest_rate, -- Average interest rate for each segment
       ROUND(AVG(amount_col), 2)                       AS avg_loan_amount,   -- Average loan amount issued
       ROUND(AVG(income_col), 2)                       AS avg_income,        -- Average income of borrowers in each category
       ROUND(AVG(percent_col), 2)                      AS avg_percent_income -- Average percentage of income allocated to loan repayment

FROM loan_data
GROUP BY grade_col, intent_col -- Grouping by loan grade and loan intent to analyze trends
ORDER BY default_rate DESC, grade_segment, intent_segment; -- Prioritize high-default categories

