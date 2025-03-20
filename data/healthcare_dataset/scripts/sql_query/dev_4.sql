-- Aggregate billing amounts at different levels using GROUPING SETS
SELECT "Hospital",                                    -- Specific hospital (can be NULL when aggregating by condition)
       "Medical Condition",                           -- Specific medical condition (can be NULL when aggregating by hospital)
       ROUND(AVG("Billing Amount"), 2) AS avg_billing -- Compute the average billing amount, rounded to 2 decimal places
FROM new_data
GROUP BY GROUPING SETS ( ("Hospital", "Medical Condition"), -- Aggregation at the level of each (Hospital, Medical Condition) pair
                         ("Hospital"),                      -- Aggregation at the hospital level (across all medical conditions)
                         ("Medical Condition")              -- Aggregation at the medical condition level (across all hospitals)
    )
ORDER BY "Hospital" NULLS FIRST, -- Ensure NULL values (aggregated medical conditions) appear first in sorting
         "Medical Condition" NULLS FIRST; -- Ensure NULL values (aggregated hospitals) appear first