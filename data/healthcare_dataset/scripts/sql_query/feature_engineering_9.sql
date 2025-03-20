SELECT "id",
       "Medical Condition",
       "Admission Type",
       "Billing Amount",
       DATEDIFF(
               'day',
               CAST("Date of Admission" AS DATE),
               CAST("Discharge Date" AS DATE)
       )       AS length_of_stay,
       CASE
           WHEN DATEDIFF(
                        'day',
                        CAST("Date of Admission" AS DATE),
                        CAST("Discharge Date" AS DATE)
                ) = 0 THEN NULL
           ELSE ROUND(
                   "Billing Amount"
                       / DATEDIFF(
                           'day',
                           CAST("Date of Admission" AS DATE),
                           CAST("Discharge Date" AS DATE)
                         ),
                   2
                )
           END AS cost_per_day
FROM new_data
ORDER BY "id";

-- columns we want to keep in the final dataset
-- id, Medical Condition, Admission Type, Billing Amount, length_of_stay, cost_per_day
-- Room number is not included in the final dataset because it is sensitive information.