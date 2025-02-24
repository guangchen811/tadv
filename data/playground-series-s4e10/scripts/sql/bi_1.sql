-- Analyze loan approval rates based on home ownership status
SELECT person_home_ownership,
       COUNT(*)                                                                          AS total_loans,
       SUM(CASE WHEN cb_person_default_on_file = 'N' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS approval_rate
FROM new_data
GROUP BY person_home_ownership
ORDER BY approval_rate DESC;