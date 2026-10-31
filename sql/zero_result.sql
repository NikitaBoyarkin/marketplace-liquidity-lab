-- Zero-result share by region population tier.
--
-- Zero results are a supply-side failure: the buyer asked and the marketplace
-- had nothing to show. It is reported by tier because the platform average hides
-- it -- thin regions carry the loss and the big cities dilute it.

SELECT
    r.population_tier,
    COUNT(*) AS searches,
    SUM(s.zero_result) AS zero_results,
    AVG(s.zero_result) AS zero_result_share
FROM read_parquet('{{RAW}}/searches.parquet') AS s
JOIN read_parquet('{{RAW}}/regions.parquet') AS r ON r.region_id = s.region_id
GROUP BY r.population_tier
ORDER BY r.population_tier
