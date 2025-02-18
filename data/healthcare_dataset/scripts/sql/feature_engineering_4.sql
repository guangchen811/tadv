WITH billing_stats AS (SELECT "Gender",
                              "Medical Condition",
                              ROUND(AVG("Billing Amount"), 2) AS avg_billing_gender_condition
                       FROM train
                       GROUP BY "Gender", "Medical Condition")
SELECT t."id",
       t."Gender",
       t."Medical Condition",
       t."Billing Amount",
       b.avg_billing_gender_condition
FROM train AS t
         LEFT JOIN billing_stats AS b
                   ON t."Gender" = b."Gender"
                       AND t."Medical Condition" = b."Medical Condition"
ORDER BY t."id";