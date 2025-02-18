WITH derived_table AS (SELECT p.person_home_ownership AS home_status,
                              p.loan_intent           AS purpose,
                              p.loan_status           AS repayment_status
                       FROM train p)
SELECT home_status                                                                        AS ownership_type,
       purpose                                                                            AS loan_purpose,
       COUNT(*)                                                                           AS loan_count,
       SUM(CASE WHEN repayment_status = 1 THEN 1 ELSE 0 END)                              AS default_count,
       ROUND(SUM(CASE WHEN repayment_status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate
FROM derived_table
GROUP BY home_status, purpose
ORDER BY default_rate DESC, ownership_type, loan_purpose;