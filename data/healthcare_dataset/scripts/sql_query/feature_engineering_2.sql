SELECT "id",
       "Age",
       CASE
           WHEN "Age" < 18 THEN 'child'
           WHEN "Age" BETWEEN 18 AND 30 THEN 'young_adult'
           WHEN "Age" BETWEEN 31 AND 50 THEN 'adult'
           WHEN "Age" BETWEEN 51 AND 64 THEN 'mid_senior'
           ELSE 'senior'
           END AS age_bucket
FROM new_data
ORDER BY "id";