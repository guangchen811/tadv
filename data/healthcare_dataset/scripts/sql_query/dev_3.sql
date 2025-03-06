SELECT "Doctor",
       COUNT(*)                        AS patient_count,
       ROUND(AVG("Billing Amount"), 2) AS avg_billing
FROM new_data
WHERE ("Hospital" = 'Powell-Wheeler' OR 'Powell-Wheeler' IS NULL)
GROUP BY "Doctor"
ORDER BY patient_count DESC LIMIT 50;