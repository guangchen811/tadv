-- Retrieve the top customers by total loan amount
SELECT id,
       SUM(loan_amnt)     AS total_loan_amount,
       AVG(loan_int_rate) AS avg_interest_rate,
       COUNT(*)           AS total_loans
FROM test
GROUP BY id
ORDER BY total_loan_amount DESC LIMIT 5;