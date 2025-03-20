-- Analyze loan approval rates based on homeownership status
SELECT person_home_ownership,                                                                            -- Homeownership status of the borrower (e.g., RENT, OWN, MORTGAGE, OTHER)
       COUNT(*)                                                                          AS total_loans, -- Total number of loan applications for each homeownership status

       -- Calculate approval rate:
       -- Count of borrowers who have not defaulted ('N') divided by total loans in the category
       SUM(CASE WHEN cb_person_default_on_file = 'N' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS approval_rate

FROM new_data
GROUP BY person_home_ownership -- Group data by homeownership status
ORDER BY approval_rate DESC; -- Sort results by approval rate in descending order