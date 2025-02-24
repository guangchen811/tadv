-- One-hot encoding for categorical features (e.g., home ownership)
SELECT *,
       CASE WHEN person_home_ownership = 'RENT' THEN 1 ELSE 0 END     AS home_ownership_rent,
       CASE WHEN person_home_ownership = 'MORTGAGE' THEN 1 ELSE 0 END AS home_ownership_mortgage,
       CASE WHEN person_home_ownership = 'OWN' THEN 1 ELSE 0 END      AS home_ownership_own
FROM new_data;