-- Segment customers based on income levels and loan grade
SELECT CASE
           WHEN train.person_income < 50000 THEN 'Low Income'
           WHEN train.person_income BETWEEN 50000 AND 100000 THEN 'Medium Income'
           ELSE 'High Income'
           END        AS income_segment,
       train.loan_grade,
       COUNT(*)       AS total_customers,
       AVG(train.loan_amnt) AS avg_loan_amount
FROM train
GROUP BY income_segment, loan_grade
ORDER BY income_segment, loan_grade;