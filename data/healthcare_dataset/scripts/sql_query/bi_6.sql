WITH condition_counts AS (SELECT "Blood Type",
                                 "Medical Condition",
                                 COUNT(*) AS condition_total,
                                 RANK()      OVER (
            PARTITION BY "Blood Type" 
            ORDER BY COUNT(*) DESC
        ) AS condition_rank
                          FROM new_data
                          GROUP BY "Blood Type", "Medical Condition")
SELECT "Blood Type",
       "Medical Condition",
       condition_total
FROM condition_counts
WHERE condition_rank <= 3
ORDER BY "Blood Type", condition_total DESC;