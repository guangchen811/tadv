SELECT
    "Medical Condition",
    COUNT(*) AS condition_count
FROM (
    SELECT *
    FROM train
    WHERE "Age" >= 65
) AS older_patients
GROUP BY "Medical Condition"
ORDER BY condition_count DESC
LIMIT 5;