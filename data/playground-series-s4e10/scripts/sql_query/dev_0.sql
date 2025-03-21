-- Loan statistics segmented by borrower income bands
SELECT CASE
           WHEN person_income < 50000 THEN 'Low Income'
           WHEN person_income BETWEEN 50000 AND 100000 THEN 'Medium Income'
           ELSE 'High Income'
           END                                             AS income_band,
       COUNT(*)                                            AS total_loans,
       ROUND(AVG(loan_amnt), 2)                            AS avg_loan_amount,
       ROUND(AVG(loan_int_rate), 2)                        AS avg_interest_rate,
       ROUND(SUM(loan_amnt) * 1.0 / SUM(person_income), 2) AS avg_loan_to_income_ratio
FROM new_data
GROUP BY income_band
ORDER BY total_loans DESC LIMIT 10;