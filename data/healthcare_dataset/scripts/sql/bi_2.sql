WITH billing_distribution AS (SELECT *,
                                     CUME_DIST() OVER (ORDER BY "Billing Amount" DESC) AS billing_cume_dist
                              FROM new_data)
SELECT "Name",
       "Insurance Provider",
       "Medical Condition",
       "Billing Amount"
FROM billing_distribution
WHERE billing_cume_dist <= 0.05
ORDER BY "Billing Amount" DESC;