-- Combine multiple risk factors to identify high-risk segments
SELECT person_home_ownership,
       loan_grade,
       COUNT(*)                                                                          AS total_loans,
       AVG(loan_int_rate)                                                                AS avg_interest_rate,
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
FROM new_data
WHERE loan_int_rate > 15.0
GROUP BY person_home_ownership, loan_grade
HAVING default_rate > 0.2
ORDER BY default_rate DESC;