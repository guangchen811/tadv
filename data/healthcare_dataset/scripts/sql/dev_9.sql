SELECT "id",
       "Name",
       "Age",
       "Medical Condition",
       CASE
           WHEN "Age" >= 65 OR "Medical Condition" IN ('Cancer', 'Diabetes') THEN 'High'
           WHEN "Age" BETWEEN 40 AND 64 THEN 'Medium'
           ELSE 'Low'
           END AS risk_level
FROM train
ORDER BY "id" ASC LIMIT 100;