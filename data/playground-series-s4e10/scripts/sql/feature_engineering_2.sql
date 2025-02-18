-- Binning employment length into categories
SELECT *,
       CASE
           WHEN person_emp_length < 2 THEN 'Junior'
           WHEN person_emp_length BETWEEN 2 AND 5 THEN 'Mid-level'
           WHEN person_emp_length > 5 THEN 'Senior'
           ELSE 'Unknown'
           END AS emp_length_category
FROM test;