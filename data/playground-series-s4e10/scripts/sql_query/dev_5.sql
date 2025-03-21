-- Simulate loan eligibility based on income and existing loans
SELECT id,
       person_income,
       SUM(loan_amnt) AS current_loans,
       CASE
           WHEN person_income > SUM(loan_amnt) * 2 THEN 'Eligible'
           ELSE 'Not Eligible'
           END        AS eligibility_status
FROM new_data
WHERE id = 43152
GROUP BY id, person_income;