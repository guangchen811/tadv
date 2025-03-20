-- Create a Common Table Expression (CTE) to compute metrics for each doctor
WITH doctor_metrics AS (SELECT "Doctor",
                               COUNT(*)                        AS patient_count, -- Count the number of patients for each doctor
                               ROUND(AVG("Billing Amount"), 2) AS avg_billing    -- Compute the average billing amount per doctor, rounded to 2 decimal places
                        FROM new_data
                        GROUP BY "Doctor" -- Group by doctor to compute these metrics at the doctor level
)

-- Main query to rank doctors based on patient count and average billing
SELECT "Doctor",
       patient_count, -- Total number of patients treated by the doctor
       avg_billing,   -- Average billing amount for the doctor

       -- Rank doctors by the number of patients in descending order (higher patient count = higher rank)
       RANK() OVER (ORDER BY patient_count DESC) AS patient_count_rank,

-- Rank doctors by average billing amount in descending order (higher billing = higher rank) RANK() OVER (ORDER BY avg_billing DESC)   AS avg_billing_rank

FROM doctor_metrics -- Use the precomputed doctor metrics from the CTE

-- Order the final results by patient count (descending), and in case of a tie, by average billing (descending)
ORDER BY patient_count DESC, avg_billing DESC

-- Limit the output to the top 10 doctors based on patient count
    LIMIT 10;
