-- Adding a feature for the ratio of loan amount to annual income
SELECT *,
       loan_amnt * 1.0 / person_income AS income_to_loan_ratio
FROM test;