WITH stays AS (SELECT "id",
                      DATEDIFF(
                              'day',
                              CAST("Date of Admission" AS DATE),
                              CAST("Discharge Date" AS DATE)
                      ) AS length_of_stay
               FROM train)
SELECT "id",
       COUNT(*)                      AS total_visits,
       SUM(length_of_stay)           AS sum_length_of_stay,
       ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay
FROM stays
GROUP BY "id"
ORDER BY "id";