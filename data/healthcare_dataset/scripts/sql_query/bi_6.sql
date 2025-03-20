-- All the columns we have:
-- "Date of Admission", "Date of Discharge", "Room Number", "Doctor", "Hospital", "Insurance Provider", "Billing Amount", "Blood Type", "Medical Condition", "Admission Type", "Medication", "id", "Name", "Test Results", "Age"

-- Create a Common Table Expression (CTE) to count medical conditions per blood type.
WITH condition_counts AS (SELECT "Blood Type",
                                 "Medical Condition",
                                 COUNT(*) AS condition_total, -- Count occurrences of each medical condition for a given blood type

                                 -- Rank conditions within each blood type based on their frequency (higher count = higher rank)
                                 RANK()      OVER (
            PARTITION BY "Blood Type"  -- Partition ranking separately for each blood type
            ORDER BY COUNT(*) DESC     -- Rank conditions by descending order of occurrence
        ) AS condition_rank
                          FROM new_data
                          GROUP BY "Blood Type", "Medical Condition" -- Aggregate by blood type and medical condition
)

-- Retrieve the top 3 most common medical conditions for each blood type
SELECT "Blood Type",
       "Medical Condition",
       condition_total -- Total occurrences of the condition within that blood type
FROM condition_counts
WHERE condition_rank <= 3 -- Only include the top 3 most frequent conditions per blood type
ORDER BY "Blood Type", condition_total DESC; -- Order results by blood type and condition frequency