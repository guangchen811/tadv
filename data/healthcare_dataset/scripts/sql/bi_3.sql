SELECT
    "Admission Type",
    SUM(CASE WHEN "Medication" = 'Aspirin'     THEN 1 ELSE 0 END) AS Aspirin_count,
    SUM(CASE WHEN "Medication" = 'Paracetamol' THEN 1 ELSE 0 END) AS Paracetamol_count,
    SUM(CASE WHEN "Medication" = 'Ibuprofen'   THEN 1 ELSE 0 END) AS Ibuprofen_count,
    SUM(CASE WHEN "Medication" = 'Penicillin'  THEN 1 ELSE 0 END) AS Penicillin_count,
    SUM(CASE WHEN "Medication" = 'Lipitor'     THEN 1 ELSE 0 END) AS Lipitor_count
FROM train
GROUP BY "Admission Type"
ORDER BY "Admission Type";