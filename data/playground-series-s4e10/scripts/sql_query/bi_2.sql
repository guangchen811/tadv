-- Track default rates for different loan intents
SELECT loan_intent,                                                                                         -- The purpose of the loan (e.g., education, medical, personal, home improvement)
       COUNT(*)                                                                          AS total_loans,    -- Total number of loans issued for each loan intent category
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END)                  AS total_defaults, -- Total number of defaults for each loan intent
       -- Calculate default rate as the fraction of defaulted loans over total loans
       SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
FROM new_data
GROUP BY loan_intent -- Group data by the loan intent category
ORDER BY default_rate DESC; -- Sort results by default rate in descending order