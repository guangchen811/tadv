-- Categorizing interest rates into tiers
SELECT *,
       CASE
           WHEN loan_int_rate < 10 THEN 'Low Rate'
           WHEN loan_int_rate BETWEEN 10 AND 20 THEN 'Medium Rate'
           ELSE 'High Rate'
           END AS interest_rate_tier
FROM new_data;