WITH segment_data AS (SELECT l.loan_grade    AS grade_category,
                             l.loan_intent   AS intent_category,
                             l.person_income AS income_level,
                             l.loan_amnt     AS loan_amount
                      FROM train l)
SELECT grade_category              AS grade_segment,
       intent_category             AS intent_segment,
       ROUND(AVG(income_level), 2) AS avg_income,
       ROUND(AVG(loan_amount), 2)  AS avg_loan_size,
       COUNT(*)                    AS total_loans
FROM segment_data
GROUP BY grade_category, intent_category
ORDER BY total_loans DESC, grade_segment, intent_segment;