-- Calculate default rates for each loan intent
SELECT loan_intent,
       COUNT(*) AS total_loans,
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
FROM test
GROUP BY loan_intent;