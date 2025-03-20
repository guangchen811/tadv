/*
This query focuses only on the frequency of loan grades. Additional attributes can be analyzed in further queries.
*/

-- Count the number of loans issued for each loan grade
SELECT loan_grade,                -- Loan credit grade (e.g., A, B, C, D, etc.), representing borrower risk level
       COUNT(*) AS count_by_grade -- Total number of loans issued for each loan grade

FROM new_data
GROUP BY loan_grade -- Grouping loans by grade to aggregate counts
ORDER BY count_by_grade DESC;
-- Sorting by loan count in descending order to highlight the most common grades

/*
Other columns that could be analyzed:
- loan_intent: Could be included to analyze loan distribution by purpose within each grade.
- loan_status: Not needed here, but could be used to compare default rates across loan grades.
- loan_int_rate: Could provide insights into interest rate variations across grades.
- person_income: Might be relevant for understanding borrower financial profiles within each grade.
- cb_person_default_on_file: Could be useful for evaluating historical default trends per grade.
*/

