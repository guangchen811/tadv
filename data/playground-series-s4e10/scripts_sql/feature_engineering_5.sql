-- Create lag features for credit history length to capture trends
SELECT *,
       cb_person_cred_hist_length - LAG(cb_person_cred_hist_length) OVER (ORDER BY id) AS credit_hist_change
FROM test;