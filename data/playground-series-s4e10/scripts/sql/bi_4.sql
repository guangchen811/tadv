-- Highlight loan intents with the highest approval volume
SELECT loan_intent,
       COUNT(*) AS approved_loans
FROM new_data
WHERE cb_person_default_on_file = 'N'
GROUP BY loan_intent
ORDER BY approved_loans DESC LIMIT 5;