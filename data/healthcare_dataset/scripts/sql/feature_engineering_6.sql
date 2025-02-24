WITH patient_admissions AS (SELECT "id",
                                   CAST("Date of Admission" AS DATE) AS admission_date
                            FROM new_data),
     admissions_ordered AS (SELECT "id",
                                   admission_date,
                                   LAG(admission_date) OVER (
            PARTITION BY "id" ORDER BY admission_date
        ) AS previous_admission_date
                            FROM patient_admissions)
SELECT "id",
       admission_date,
       CASE
           WHEN previous_admission_date IS NULL THEN 0
           ELSE DATEDIFF('day', previous_admission_date, admission_date)
           END AS days_since_last_admission
FROM admissions_ordered
ORDER BY "id", admission_date;