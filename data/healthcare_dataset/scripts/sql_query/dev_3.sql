SELECT "Doctor",                                         -- Grouping by "Doctor", but future steps might involve analyzing trends by "Admission Type" or "Medical Condition".

       COUNT(*)                        AS patient_count, -- Counting patients per doctor; further analysis could examine the relationship with "Room Number" assignments.

       ROUND(AVG("Billing Amount"), 2) AS avg_billing    -- Average billing per doctor; a next step could be to compare costs based on "Insurance Provider" coverage.

FROM new_data
WHERE ("Hospital" = 'Powell-Wheeler' OR 'Powell-Wheeler' IS NULL) -- Filtering for a specific hospital; future studies might explore how "Test Results" impact billing.

GROUP BY "Doctor" -- Aggregating by doctor; another possible approach could be grouping by "Date of Admission" to track billing trends over time.

ORDER BY patient_count DESC -- Sorting doctors by patient volume; an alternative ranking could involve sorting by "Discharge Date" to assess case turnover.

    LIMIT 50; -- Limiting results to 50 doctors; next steps might involve filtering by "Medication" usage patterns to identify prescription tendencies.