-- Age is personal information that can be used to identify individuals.
-- We should not include it in the final report and could not use it together with other columns like, Name, Room Number, etc.

SELECT "Medical Condition",
       COUNT(*) AS condition_count
FROM (SELECT *
      FROM new_data
      WHERE "Age" >= 65) AS older_patients
GROUP BY "Medical Condition"
ORDER BY condition_count DESC LIMIT 5;