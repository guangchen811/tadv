-- Analyze loan distribution across different income bands
SELECT CASE
           WHEN person_income < 50000 THEN 'Low Income' -- Borrowers earning less than 50K
           WHEN person_income BETWEEN 50000 AND 100000 THEN 'Medium Income' -- Borrowers earning between 50K and 100K
           ELSE 'High Income' -- Borrowers earning above 100K
           END            AS income_band,

       COUNT(*)           AS total_loans,      -- Total number of loans issued within each income band

       -- Decided to include loan amount because it’s useful to see if different income groups borrow significantly different amounts
       AVG(loan_amnt)     AS avg_loan_amount,  -- Average loan amount issued for each income segment

       -- Including interest rate to check if higher income borrowers get better rates
       AVG(loan_int_rate) AS avg_interest_rate -- Average interest rate for borrowers in each segment

FROM new_data

-- Using income bands instead of raw income values to make it easier to analyze patterns
GROUP BY income_band

-- Sorting by total loans so that the most common income groups appear first
ORDER BY total_loans DESC;

/*
If needed, adding person_home_ownership or employment length might reveal more insights.
Might revisit this later to see how income bands relate to default rates.
*/