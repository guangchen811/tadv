-- Default rate and loan statistics by employment length
SELECT person_emp_length                                                             AS emp_length,
       COUNT(*)                                                                      AS total_loans,
       SUM(CASE WHEN loan_status = 1 THEN 1 ELSE 0 END)                              AS total_defaults,
       ROUND(SUM(CASE WHEN loan_status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate,
       AVG(loan_amnt)                                                                AS avg_loan_amount,
       AVG(loan_int_rate)                                                            AS avg_interest_rate
FROM new_data
GROUP BY person_emp_length
ORDER BY default_rate DESC;