-- Combine multiple risk factors to identify high-risk segments
SELECT person_home_ownership,                                                                                  -- The type of home ownership (e.g., RENT, OWN, MORTGAGE, OTHER), which may influence financial stability
       loan_grade,                                                                                             -- Loan credit grade (e.g., A, B, C, D), which represents the borrower's creditworthiness

       COUNT(*)                                                                          AS total_loans,       -- Total number of loans in each home ownership and loan grade segment

       AVG(loan_int_rate)                                                                AS avg_interest_rate, -- Average interest rate for loans in this category

       -- Calculate default rate: the percentage of loans that defaulted within each category
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate

FROM new_data
WHERE loan_int_rate > 15.0 -- Filter loans with a high interest rate, assuming these are riskier
GROUP BY person_home_ownership, loan_grade -- Group by home ownership type and loan grade
HAVING default_rate > 0.2 -- Consider only segments where the default rate is higher than 20%
ORDER BY default_rate DESC; -- Sort results to show the highest default rate first