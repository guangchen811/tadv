SELECT "Hospital",
       "Admission Type",
       ROUND(
               COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY "Hospital"),
               3
       ) AS admission_ratio
FROM new_data
GROUP BY "Hospital", "Admission Type"
ORDER BY "Hospital", admission_ratio DESC;