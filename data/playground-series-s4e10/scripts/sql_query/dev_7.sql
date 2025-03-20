-- Aggregate loan statistics for home ownership categories
SELECT person_home_ownership                            AS home_ownership, -- Type of home ownership (e.g., RENT, OWN, MORTGAGE)

       COUNT(*)                                         AS total_loans,    -- Total number of loans issued within each home ownership category

       -- Checking if loan amounts significantly differ between homeowners and renters
       AVG(loan_amnt)                                   AS avg_loan_amount,

       -- Analyzing whether homeownership status affects the interest rate borrowers receive
       AVG(loan_int_rate)                               AS avg_interest_rate,

       -- Counting total loan defaults to examine risk distribution among home ownership types
       SUM(CASE WHEN loan_status = 1 THEN 1 ELSE 0 END) AS total_defaults

FROM new_data

-- Grouping by home ownership to analyze loan patterns within each category
GROUP BY person_home_ownership

-- Sorting by total loans so the most common home ownership types appear first
ORDER BY total_loans DESC;

/*
While writing this query, I thought about adding a default rate calculation
(e.g., total_defaults / total_loans) to see if homeowners default less than renters.
Would also be interesting to check loan_percent_income to see if renters allocate a higher portion of income to loans.
Might revisit later if a deeper risk analysis is needed.
*/