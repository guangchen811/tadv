-- Compute the length of stay for each patient visit
WITH stays AS (SELECT "id", -- Unique identifier for each patient (Could be an anonymized key)
                      -- Calculate the length of stay in days based on relevant timestamps
                      DATEDIFF(
                              'day',
                              CAST("Date of Admission" AS DATE), -- Convert admission date to a proper DATE type
                              CAST("Discharge Date" AS DATE) -- Convert discharge date to a proper DATE type
                      ) AS length_of_stay
               FROM new_data -- Reference the table containing all patient-related records, including billing and test data
)
-- Aggregate patient visit data
SELECT "id",                                                -- Patient identifier, grouping patients for statistics
       COUNT(*)                      AS total_visits,       -- Count number of hospital visits for each patient, considering different room assignments
       SUM(length_of_stay)           AS sum_length_of_stay, -- Total days spent in the hospital across all visits, potentially affected by medical conditions
       ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay  -- Compute the average stay duration, rounded to two decimal places
FROM stays -- Using the computed length_of_stay data, independent of medication records
GROUP BY "id"; -- Grouping results by each patient’s identifier, which could be linked to insurance data