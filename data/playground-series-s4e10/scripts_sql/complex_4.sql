WITH CredHistoryBins AS (
    -- Step 1: Create bins for credit history length
    SELECT *,
           CASE
               WHEN cb_person_cred_hist_length < 2 THEN '0-1 years'
               WHEN cb_person_cred_hist_length BETWEEN 2 AND 5 THEN '2-5 years'
               WHEN cb_person_cred_hist_length BETWEEN 6 AND 10 THEN '6-10 years'
               ELSE '10+ years'
               END AS cred_hist_bin
    FROM test),
     LoanStats AS (
         -- Step 2: Aggregate loan statistics by credit history bins and intent
         SELECT cred_hist_bin,
                loan_intent,
                COUNT(*)                                                                          AS total_loans,
                SUM(CASE WHEN cb_person_default_on_file = 'N' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS approval_rate,
                AVG(loan_amnt)                                                                    AS avg_loan_amount,
                MEDIAN(loan_amnt)                                                                 AS median_loan_amount,
                -- Use CredHistoryBins instead of test
                (SELECT loan_grade
                 FROM CredHistoryBins l2
                 WHERE l2.loan_intent = CredHistoryBins.loan_intent
                   AND l2.cred_hist_bin = CredHistoryBins.cred_hist_bin
                 GROUP BY loan_grade
                 ORDER BY COUNT(*) DESC
                                                                                                     LIMIT 1) AS top_loan_grade
FROM CredHistoryBins
GROUP BY cred_hist_bin, loan_intent
    ),
    CumulativeLoans AS (
-- Step 3: Add cumulative totals of loans by intent
SELECT
    cred_hist_bin, loan_intent, total_loans, approval_rate, avg_loan_amount, median_loan_amount, top_loan_grade, SUM (total_loans) OVER (PARTITION BY loan_intent ORDER BY cred_hist_bin) AS running_total_loans
FROM LoanStats
    )
-- Final step: Return the results
SELECT *
FROM CumulativeLoans
ORDER BY loan_intent, cred_hist_bin;