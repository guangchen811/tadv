-- Find the top 5% of billing amounts in the dataset.
WITH billing_distribution AS (SELECT *,
                                     CUME_DIST() OVER (ORDER BY "Billing Amount" DESC) AS billing_cume_dist
                              FROM new_data)
SELECT "Name",
       "Medical Condition",
       "Billing Amount"
-- TODO: We should also Insurance Provider in the feature.
FROM billing_distribution
WHERE billing_cume_dist <= 0.05
ORDER BY "Billing Amount" DESC;