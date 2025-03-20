-- Select relevant patient details
SELECT "id",
       "Name",
       "Gender",
       "Insurance Provider",
       "Admission Type",
       "Medical Condition"
FROM new_data
WHERE
  -- Filter by insurance provider (If 'Medicare' is NULL, include all records)
    ('Medicare' IS NULL OR "Insurance Provider" = 'Medicare')

  -- Filter by gender (If 'Female' is NULL, include all records)
  AND ('Female' IS NULL OR "Gender" = 'Female')

  -- Filter by admission type (If 'Emergency' is NULL, include all records)
  AND ('Emergency' IS NULL OR "Admission Type" = 'Emergency')

-- Order the results by ID in ascending order
ORDER BY "id" ASC

-- Limit output to the first 100 records
    LIMIT 100;

-- Other columns available in the dataset:
-- "Date of Admission", "Date of Discharge", "Room Number", "Doctor"
-- "Hospital", "Billing Amount", "Blood Type"
-- "Medical Condition", "Admission Type", "Medication", "Test Results", "Age"