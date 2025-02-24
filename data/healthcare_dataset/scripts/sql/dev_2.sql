SELECT "id",
       "Name",
       "Date of Admission",
       "Test Results",
       "Medical Condition",
       "Billing Amount"
FROM new_data
WHERE CAST("Date of Admission" AS DATE) BETWEEN '2020-01-01' AND '2022-12-31'
  AND "Test Results" = 'Abnormal'
ORDER BY "Date of Admission" ASC;