-- Segment customers based on income levels and loan grade
SELECT CASE
           WHEN person_income < 50000 THEN 'Low Income'
           WHEN person_income BETWEEN 50000 AND 100000 THEN 'Medium Income'
           ELSE 'High Income'
           END        AS income_segment,  -- Categorizing borrowers based on income levels

       loan_grade,                        -- Credit grade of the loan, which reflects borrower risk

       COUNT(*)       AS total_customers, -- Number of customers in each income and loan grade category

       AVG(loan_amnt) AS avg_loan_amount  -- Average loan amount issued for each segment

FROM new_data
GROUP BY income_segment, loan_grade
ORDER BY income_segment, loan_grade;

-- Other columns in the dataset, such as person_age, person_home_ownership, and person_emp_length,
-- could provide additional insights but are not directly needed for this segmentation.
-- Similarly, loan_int_rate and loan_status might be useful for further analysis on risk and repayment behavior.