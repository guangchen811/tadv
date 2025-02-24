SELECT new_data.loan_grade,
       new_data.loan_intent,
       COUNT(*)                                                                            AS total_loans,
       SUM(CASE WHEN new_data.loan_status = 1 THEN 1 ELSE 0 END)                              AS total_defaults,
       ROUND(SUM(CASE WHEN new_data.loan_status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate_percentage,
       ROUND(AVG(new_data.loan_amnt), 2)                                                      AS avg_loan_amount,
       ROUND(AVG(new_data.person_income), 2)                                                  AS avg_borrower_income,
       ROUND(AVG(new_data.loan_percent_income), 2)                                            AS avg_loan_percent_income
FROM new_data
GROUP BY loan_grade,
         loan_intent
ORDER BY default_rate_percentage DESC,
         loan_grade,
         loan_intent;