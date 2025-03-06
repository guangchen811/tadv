-- Track default rates for different loan intents
SELECT loan_intent,
       COUNT(*)                                                                          AS total_loans,
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END)                  AS total_defaults,
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
FROM new_data
GROUP BY loan_intent
ORDER BY default_rate DESC;