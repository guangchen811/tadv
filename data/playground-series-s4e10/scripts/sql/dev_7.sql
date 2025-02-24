-- Aggregate loan statistics for home ownership categories
SELECT person_home_ownership                            AS home_ownership,
       COUNT(*)                                         AS total_loans,
       AVG(loan_amnt)                                   AS avg_loan_amount,
       AVG(loan_int_rate)                               AS avg_interest_rate,
       SUM(CASE WHEN loan_status = 1 THEN 1 ELSE 0 END) AS total_defaults
FROM new_data
GROUP BY person_home_ownership
ORDER BY total_loans DESC;