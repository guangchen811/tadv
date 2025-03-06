-- Adding a derived feature for default probability based on intent and income
SELECT *,
       CASE loan_intent
           WHEN 'DEBTCONSOLIDATION' THEN 0.25
           WHEN 'EDUCATION' THEN 0.15
           WHEN 'VENTURE' THEN 0.35
           ELSE 0.20
           END +
       CASE
           WHEN person_income < 50000 THEN 0.1
           WHEN person_income BETWEEN 50000 AND 100000 THEN 0.05
           ELSE -0.05
           END AS default_probability_score
FROM new_data;