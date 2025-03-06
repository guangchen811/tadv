SELECT "id",
       SUM(CASE WHEN "Medication" = 'Aspirin' THEN 1 ELSE 0 END)     AS med_aspirin,
       SUM(CASE WHEN "Medication" = 'Paracetamol' THEN 1 ELSE 0 END) AS med_paracetamol,
       SUM(CASE WHEN "Medication" = 'Ibuprofen' THEN 1 ELSE 0 END)   AS med_ibuprofen,
       SUM(CASE WHEN "Medication" = 'Penicillin' THEN 1 ELSE 0 END)  AS med_penicillin,
       SUM(CASE WHEN "Medication" = 'Lipitor' THEN 1 ELSE 0 END)     AS med_lipitor
FROM new_data
GROUP BY "id"
ORDER BY "id";