SELECT "id",
       SUM(CASE WHEN "Medication" = 'Aspirin' THEN 1 ELSE 0 END)     AS med_aspirin,
       SUM(CASE WHEN "Medication" = 'Paracetamol' THEN 1 ELSE 0 END) AS med_paracetamol,
       SUM(CASE WHEN "Medication" = 'Ibuprofen' THEN 1 ELSE 0 END)   AS med_ibuprofen,
       SUM(CASE WHEN "Medication" = 'Penicillin' THEN 1 ELSE 0 END)  AS med_penicillin,
       SUM(CASE WHEN "Medication" = 'Lipitor' THEN 1 ELSE 0 END)     AS med_lipitor
FROM new_data
GROUP BY "id"
ORDER BY "id" LIMIT 10;
-- Output:
-- id,med_aspirin,med_paracetamol,med_ibuprofen,med_penicillin,med_lipitor

-- The dataset may contain multiple records for each patient visit.
-- We hard code the medication names in the query. We could improve this by dynamically fetching the medication names from the dataset or ask the doctor for a list of common medications.