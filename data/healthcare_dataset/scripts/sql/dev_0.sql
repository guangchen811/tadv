SELECT "id",
       "Name",
       "Age",
       "Gender",
       "Blood Type",
       "Medical Condition",
       "Date of Admission",
       "Doctor",
       "Hospital",
       "Insurance Provider",
       "Billing Amount",
       "Room Number",
       "Admission Type",
       "Discharge Date",
       "Medication",
       "Test Results"
FROM new_data
WHERE ("Medical Condition" ILIKE 'Asthma' OR "Admission Type" ILIKE 'Emergency')
  AND "Age" BETWEEN 30 AND 70
  AND "Billing Amount" > 5000
ORDER BY "Billing Amount" DESC LIMIT 5
OFFSET 0;