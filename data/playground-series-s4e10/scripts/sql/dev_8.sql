-- Analyze loan distribution and average interest rates by intent
SELECT loan_intent                                         AS intent_category,
       COUNT(*)                                            AS total_loans,
       AVG(loan_amnt)                                      AS avg_loan_amount,
       ROUND(AVG(loan_int_rate), 2)                        AS avg_interest_rate,
       ROUND(SUM(loan_amnt) * 1.0 / SUM(person_income), 2) AS avg_loan_to_income_ratio
FROM new_data
GROUP BY loan_intent
ORDER BY total_loans DESC;