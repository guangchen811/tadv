-- Step 1: Filter loans with specific conditions
WITH FilteredLoans AS (
    SELECT *
    FROM test
    WHERE loan_grade >= 'C'  -- Include loans with grade C or higher
      AND loan_int_rate > 15.0  -- Include loans with interest rate greater than 15%
),

-- Step 2: Categorize income into bands
IncomeBands AS (
    SELECT *,
           CASE
               WHEN person_income < 50000 THEN 'LOW'  -- Low-income group
               WHEN person_income BETWEEN 50000 AND 100000 THEN 'MEDIUM'  -- Medium-income group
               ELSE 'HIGH'  -- High-income group
           END AS income_band  -- Derived column for income categorization
    FROM FilteredLoans
),

-- Step 3: Rank loans within each intent by loan_percent_income
RankedLoans AS (
    SELECT *,
           RANK() OVER (PARTITION BY loan_intent ORDER BY loan_percent_income DESC) AS loan_rank  -- Rank loans by percentage of income
    FROM IncomeBands
    WHERE cb_person_cred_hist_length > 3  -- Include only loans with credit history length greater than 3 years
)

-- Step 4: Aggregate and calculate statistics for each loan intent
SELECT loan_intent,
       COUNT(*)           AS total_loans,  -- Total number of loans for the intent
       AVG(loan_amnt)     AS avg_loan_amount,  -- Average loan amount
       AVG(loan_int_rate) AS avg_interest_rate,  -- Average interest rate
       MAX(loan_rank)     AS max_rank_in_intent  -- Maximum rank within the intent
FROM RankedLoans
GROUP BY loan_intent  -- Group results by loan intent
ORDER BY loan_intent;  -- Sort results by loan intent