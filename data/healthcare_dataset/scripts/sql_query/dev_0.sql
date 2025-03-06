SELECT *
FROM new_data
WHERE ("Medical Condition" ILIKE 'Asthma' OR "Admission Type" ILIKE 'Emergency')
  AND "Age" BETWEEN 30 AND 70
  AND "Billing Amount" > 5000
ORDER BY "Billing Amount" DESC LIMIT 5
OFFSET 0;