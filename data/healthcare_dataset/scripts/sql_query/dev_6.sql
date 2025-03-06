SELECT "id",
       "Name",
       "Gender",
       "Insurance Provider",
       "Admission Type",
       "Medical Condition"
FROM new_data
WHERE ('Medicare' IS NULL OR "Insurance Provider" = 'Medicare')
  AND ('Female' IS NULL OR "Gender" = 'Female')
  AND ('Emergency' IS NULL OR "Admission Type" = 'Emergency')
ORDER BY "id" ASC LIMIT 100;