-- Count and calculate default rate by loan grade
SELECT t.loan_grade,
       COUNT(*)                                                                        AS total_loans,
       SUM(CASE WHEN t.loan_status = 1 THEN 1 ELSE 0 END)                              AS total_defaults,
       ROUND(SUM(CASE WHEN t.loan_status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate
FROM new_data AS t
GROUP BY t.loan_grade
ORDER BY total_loans DESC;