-- Cohort liquidity by category, computed directly from the raw parquet log.
--
-- Cohort = listings published inside the first {{WINDOW_DAYS}} days of the data
-- window. It is fixed by publish time, never by "still active at the end of the
-- window": conditioning on survival flatters every liquidity number.
--
-- A contact is one distinct (buyer_id, listing_id) pair, never a row count.
-- Every cohort listing is observed for a full {{WINDOW_DAYS}} days, so contacts
-- and views outside that window are excluded.

WITH cohort AS (
    SELECT listing_id, object_id, category_id, publish_day
    FROM read_parquet('{{RAW}}/listings.parquet')
    WHERE publish_day < {{WINDOW_DAYS}}
),

duplicated_objects AS (
    SELECT object_id
    FROM read_parquet('{{RAW}}/listings.parquet')
    GROUP BY object_id
    HAVING COUNT(*) > 1
),

contacted AS (
    SELECT DISTINCT c.listing_id
    FROM read_parquet('{{RAW}}/contacts.parquet') AS c
    JOIN cohort AS ch ON ch.listing_id = c.listing_id
    WHERE c.day < ch.publish_day + {{WINDOW_DAYS}}
),

viewed AS (
    SELECT DISTINCT v.listing_id
    FROM read_parquet('{{RAW}}/views.parquet') AS v
    JOIN cohort AS ch ON ch.listing_id = v.listing_id
    WHERE v.day < ch.publish_day + {{WINDOW_DAYS}}
)

SELECT
    ch.category_id,
    COUNT(*) AS cohort_size,
    AVG(CASE WHEN ct.listing_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS listings_with_contact,
    AVG(CASE WHEN vw.listing_id IS NULL THEN 1.0 ELSE 0.0 END) AS dead_listing_share,
    AVG(CASE WHEN d.object_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS duplicate_share
FROM cohort AS ch
LEFT JOIN contacted AS ct ON ct.listing_id = ch.listing_id
LEFT JOIN viewed AS vw ON vw.listing_id = ch.listing_id
LEFT JOIN duplicated_objects AS d ON d.object_id = ch.object_id
GROUP BY ch.category_id
ORDER BY ch.category_id
