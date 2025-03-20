-- Compute the average billing amount for each (Gender, Medical Condition) pair
WITH billing_stats AS (SELECT "Gender",
                              "Medical Condition",
                              ROUND(AVG("Billing Amount"), 2) AS avg_billing_gender_condition -- Calculate average billing per gender-condition group
                       FROM new_data
                       GROUP BY "Gender", "Medical Condition" -- Group by gender and medical condition to compute aggregated values
)

-- Retrieve patient details along with their corresponding average billing amount for the same gender and condition
SELECT t."id",                        -- Patient ID
       t."Gender",                    -- Patient's gender
       t."Medical Condition",         -- Patient's medical condition
       t."Billing Amount",            -- Patient's actual billing amount
       b.avg_billing_gender_condition -- Corresponding average billing amount for the same gender-condition group

FROM new_data AS t

-- Perform a LEFT JOIN to match each patient's gender and condition with the precomputed average billing amount
         LEFT JOIN billing_stats AS b
                   ON t."Gender" = b."Gender"
                       AND t."Medical Condition" = b."Medical Condition"

-- Order the results by patient ID for structured output
ORDER BY t."id";