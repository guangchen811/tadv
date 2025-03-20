-- Compute the length of stay for each patient visit
WITH stays AS (SELECT "id",               -- Patient ID
                      DATEDIFF(
                              'day',
                              CAST("Date of Admission" AS DATE), -- Convert admission date to DATE type
                              CAST("Discharge Date" AS DATE) -- Convert discharge date to DATE type
                      ) AS length_of_stay -- Calculate the duration of stay in days
               FROM new_data)

-- Aggregate patient visit data
SELECT "id",                                                -- Patient ID
       COUNT(*)                      AS total_visits,       -- Total number of visits per patient
       SUM(length_of_stay)           AS sum_length_of_stay, -- Total number of days spent in hospital across visits
       ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay  -- Average length of stay per patient, rounded to 2 decimal places
FROM stays
GROUP BY "id" -- Group by patient ID to calculate stats for each individual