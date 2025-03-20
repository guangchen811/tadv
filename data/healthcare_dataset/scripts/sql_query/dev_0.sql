SELECT * EXCLUDE "Gender", "Room Number"

FROM new_data
WHERE
  -- Filter for patients with 'Asthma' as a medical condition OR 'Emergency' as an admission type
    ("Medical Condition" ILIKE 'Asthma' OR "Admission Type" ILIKE 'Emergency')

  -- Ensure the patient's age is between 30 and 70 (inclusive)
  AND "Age" BETWEEN 30 AND 70

  -- Include only cases where the billing amount exceeds 5000
  AND "Billing Amount" > 5000

-- Order results by highest billing amount first
ORDER BY "Billing Amount" DESC

-- Limit the result to the top 5 entries
    LIMIT 5

-- Start from the first row (useful for pagination)
OFFSET 0;

-- The Gender column is not included in the final report as it is considered sensitive information.