WITH billing_rank AS (SELECT "id",
                             "Name",
                             "Medical Condition",
                             "Billing Amount",
                             CUME_DIST() OVER (ORDER BY "Billing Amount" DESC) AS billing_percentile
                      FROM train)
SELECT "id",
       "Name",
       "Medical Condition",
       "Billing Amount"
FROM billing_rank
WHERE billing_percentile <= 0.05
ORDER BY "Billing Amount" DESC;