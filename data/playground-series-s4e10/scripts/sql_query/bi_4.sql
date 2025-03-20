-- Highlight loan intents with the highest approval volume
SELECT loan_intent,               -- Purpose of the loan (e.g., education, medical, personal, home improvement)

       COUNT(*) AS approved_loans -- Total number of loans approved for each loan intent

FROM new_data
WHERE cb_person_default_on_file = 'N' -- Filter only non-default borrowers, assuming they are more likely to get loan approval
GROUP BY loan_intent -- Group by loan purpose
ORDER BY approved_loans DESC -- Sort in descending order to show loan intents with the highest approvals first
    LIMIT 5;
-- Show only the top 5 loan intents with the highest approval volume
-- This query focuses on loan intents rather than borrower demographics. person_age could be included in a deeper analysis of approval trends.
-- While income can influence loan approval, this query only looks at overall approval counts, not the financial profile of approved borrowers, thus excluding person_income.