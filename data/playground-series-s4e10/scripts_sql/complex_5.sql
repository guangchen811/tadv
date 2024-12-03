WITH IncomeBands AS (
    -- Step 1: Create income bands for segmentation
    SELECT *,
           CASE
               WHEN person_income < 50000 THEN 'Low Income'
               WHEN person_income BETWEEN 50000 AND 100000 THEN 'Medium Income'
               ELSE 'High Income'
               END AS income_band
    FROM test),
     CredHistoryBins AS (
         -- Step 2: Create credit history bins
         SELECT *,
                CASE
                    WHEN cb_person_cred_hist_length < 2 THEN '0-1 years'
                    WHEN cb_person_cred_hist_length BETWEEN 2 AND 5 THEN '2-5 years'
                    WHEN cb_person_cred_hist_length BETWEEN 6 AND 10 THEN '6-10 years'
                    ELSE '10+ years'
                    END AS cred_hist_bin
         FROM IncomeBands),
     AggregatedStats AS (
         -- Step 3: Aggregate statistics for each income band and credit history bin
         SELECT income_band,
                cred_hist_bin,
                loan_intent,
                COUNT(*)                                                                                  AS total_loans,
                SUM(CASE WHEN cb_person_default_on_file = 'N' THEN 1 ELSE 0 END) * 1.0 /
                COUNT(*)                                                                                  AS approval_rate,
                AVG(loan_amnt)                                                                            AS avg_loan_amount,
                SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN loan_amnt ELSE 0 END) * 1.0 /
                COUNT(*)                                                                                  AS avg_defaulted_amount
         FROM CredHistoryBins
         GROUP BY income_band, cred_hist_bin, loan_intent),
     PivotedStats AS (
         -- Step 4: Pivot loan statistics by intent
         SELECT income_band,
                cred_hist_bin,
                MAX(CASE WHEN loan_intent = 'HOMEIMPROVEMENT' THEN approval_rate ELSE NULL END)   AS home_improvement_approval_rate,
                MAX(CASE WHEN loan_intent = 'PERSONAL' THEN approval_rate ELSE NULL END)          AS personal_approval_rate,
                MAX(CASE WHEN loan_intent = 'VENTURE' THEN approval_rate ELSE NULL END)           AS venture_approval_rate,
                MAX(CASE
                        WHEN loan_intent = 'HOMEIMPROVEMENT' THEN avg_loan_amount
                        ELSE NULL END)                                                            AS home_improvement_avg_loan,
                MAX(CASE WHEN loan_intent = 'PERSONAL' THEN avg_loan_amount ELSE NULL END)        AS personal_avg_loan,
                MAX(CASE WHEN loan_intent = 'VENTURE' THEN avg_loan_amount ELSE NULL END)         AS venture_avg_loan
         FROM AggregatedStats
         GROUP BY income_band, cred_hist_bin)
-- Final step: Return the results
SELECT *
FROM PivotedStats
ORDER BY income_band, cred_hist_bin;