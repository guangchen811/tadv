-- We use the GROUPING SETS clause to calculate the average billing amount for each unique combination of "Insurance Provider" and "Medical Condition".
-- Date of Admission, Discharge Date, Hospital, Room Number, and Doctor are not considered in this query but could be useful in future analyses to find some patterns.
SELECT "Insurance Provider",
       "Medical Condition",
       ROUND(AVG("Billing Amount"), 2) AS avg_billing
FROM new_data
GROUP BY GROUPING SETS ( ("Insurance Provider", "Medical Condition"),
                         ("Insurance Provider"),
                         ("Medical Condition")
    )
ORDER BY "Insurance Provider", "Medical Condition";
