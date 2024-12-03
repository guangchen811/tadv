WITH OverallStats AS (
    SELECT AVG(loan_amnt) AS overall_avg_loan
    FROM test
),
FilteredCategories AS (
    SELECT loan_intent,
           AVG(loan_amnt) AS avg_loan_amount
    FROM test
    GROUP BY loan_intent
    HAVING AVG(loan_amnt) > (SELECT overall_avg_loan FROM OverallStats)  -- Filter loan_intent with avg_loan_amount > overall average
),
CategoryDetails AS (
    SELECT t.loan_intent,
           COUNT(*) AS total_loans,
           MIN(t.loan_int_rate) AS min_interest_rate,
           MAX(t.loan_int_rate) AS max_interest_rate,
           MEDIAN(t.person_emp_length) AS median_emp_length,
           SUM(CASE WHEN t.cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate,
           COUNT(DISTINCT CASE
                            WHEN t.person_income < 50000 THEN 'LOW'
                            WHEN t.person_income BETWEEN 50000 AND 100000 THEN 'MEDIUM'
                            ELSE 'HIGH'
                          END) AS distinct_income_bands
    FROM test t
    WHERE t.loan_intent IN (SELECT loan_intent FROM FilteredCategories)  -- Filter relevant loan_intents
    GROUP BY t.loan_intent
),
FinalResult AS (
    SELECT loan_intent,
           total_loans,
           min_interest_rate,
           max_interest_rate,
           median_emp_length,
           CASE
               WHEN default_rate > 0.5 THEN 'Yes'
               ELSE 'No'
           END AS high_default_rate,  -- Flag if more than 50% have default on file
           distinct_income_bands
    FROM CategoryDetails
    WHERE distinct_income_bands >= 3  -- Only include categories with at least 3 income bands
)
SELECT *
FROM FinalResult
ORDER BY loan_intent;  -- Sort by loan_intent