-- Encoding loan grades as numeric values for modeling
SELECT *,
       CASE
           WHEN loan_grade = 'A' THEN 1
           WHEN loan_grade = 'B' THEN 2
           WHEN loan_grade = 'C' THEN 3
           WHEN loan_grade = 'D' THEN 4
           WHEN loan_grade = 'E' THEN 5
           WHEN loan_grade = 'F' THEN 6
           WHEN loan_grade = 'G' THEN 7
           ELSE NULL
           END AS grade_numeric
FROM new_data;