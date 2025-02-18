-- Count loans by grade
SELECT loan_grade,
       COUNT(*) AS count_by_grade
FROM test
GROUP BY loan_grade
ORDER BY count_by_grade DESC;