WITH billing_rank AS (SELECT *,
                             CUME_DIST() OVER (ORDER BY "Billing Amount" DESC) AS billing_percentile
                      FROM new_data)
SELECT "id",
       "Name",
       "Medical Condition",
       "Billing Amount"
FROM billing_rank
WHERE billing_percentile <= 0.05
ORDER BY "Billing Amount" DESC;