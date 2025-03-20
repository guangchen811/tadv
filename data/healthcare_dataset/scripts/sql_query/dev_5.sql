WITH billing_rank AS (SELECT *,
                             CUME_DIST() OVER (ORDER BY "Billing Amount" DESC) AS billing_percentile
                      FROM new_data -- Dataset contains patient details, including "Insurance Provider", "Doctor", and "Room Number", which might be analyzed in future steps.
)

SELECT "id",                -- Unique patient identifier, could be further analyzed alongside "Date of Admission" for trends in high-cost cases.

       "Name",              -- Patient name, possibly useful in later analyses when linking to specific "Hospital" records.

       "Medical Condition", -- The diagnosed condition; next steps could involve exploring its relationship with "Medication" prescriptions.

       "Billing Amount"     -- The total cost for treatment; subsequent analyses might consider its correlation with "Test Results" or "Admission Type".

FROM billing_rank
WHERE billing_percentile <= 0.05 -- Filtering for the top 5% most expensive cases; future refinements may involve examining "Insurance Provider" coverage.
ORDER BY "Billing Amount" DESC; -- Sorting by highest charges; another approach could be ranking by "Doctor" to assess cost variability.