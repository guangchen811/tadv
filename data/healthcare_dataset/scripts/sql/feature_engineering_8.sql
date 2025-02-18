WITH patient_bills AS (SELECT "id",
                              CAST("Date of Admission" AS DATE) AS admission_date,
                              "Billing Amount"                  AS billing
                       FROM train),
     bills_ordered AS (SELECT "id",
                              admission_date,
                              billing,
                              AVG(billing) OVER (
            PARTITION BY "id"
            ORDER BY admission_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_billing_3_visits
                       FROM patient_bills)
SELECT "id",
       admission_date,
       billing,
       ROUND(rolling_avg_billing_3_visits, 2) AS rolling_avg_billing_3_visits
FROM bills_ordered
ORDER BY "id", admission_date;