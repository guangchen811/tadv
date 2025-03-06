-- Segment customers based on income levels and loan grade
SELECT CASE
           WHEN person_income < 50000 THEN 'Low Income'
           WHEN person_income BETWEEN 50000 AND 100000 THEN 'Medium Income'
           ELSE 'High Income'
           END        AS income_segment,
       loan_grade,
       COUNT(*)       AS total_customers,
       AVG(loan_amnt) AS avg_loan_amount
FROM new_data
GROUP BY income_segment, loan_grade
ORDER BY income_segment, loan_grade;