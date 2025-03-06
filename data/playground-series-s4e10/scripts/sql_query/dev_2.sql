-- Aggregate loan statistics for a category (e.g., intent or home ownership)
SELECT loan_intent        AS category,
       COUNT(*)           AS total_loans,
       AVG(loan_amnt)     AS avg_loan_amount,
       AVG(loan_int_rate) AS avg_interest_rate
FROM new_data
GROUP BY loan_intent
ORDER BY total_loans DESC;