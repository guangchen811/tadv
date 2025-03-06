-- Create a temporary view to calculate the average billing amount per gender and medical condition
CREATE
TEMP VIEW billing_stats AS
SELECT "Medical Condition",
       "Gender",
       ROUND(AVG("Billing Amount"), 2) AS avg_billing_gender_condition
FROM new_data
GROUP BY "Gender", "Medical Condition";

-- Perform a LEFT JOIN to attach the computed averages to the original dataset
SELECT t."id",
       t."Medical Condition",
       t."Billing Amount",
       b.avg_billing_gender_condition
FROM new_data AS t
         LEFT JOIN billing_stats AS b
                   ON t."Gender" = b."Gender"
                       AND t."Medical Condition" = b."Medical Condition"
ORDER BY t."id";