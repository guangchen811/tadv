SELECT "Hospital",
       "Medical Condition",
       ROUND(AVG("Billing Amount"), 2) AS avg_billing
FROM new_data
GROUP BY GROUPING SETS ( ("Hospital", "Medical Condition"),
                         ("Hospital"),
                         ("Medical Condition")
    )
ORDER BY "Hospital" NULLS FIRST, "Medical Condition" NULLS FIRST;