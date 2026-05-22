-- ============================================================
-- CONSUMER BEHAVIOUR PLATFORM — SQL QUERIES
-- Database: consumer_behaviour_db
-- ============================================================

-- Q1_platform_by_age
SELECT 
    age_group,
    primary_platform,
    COUNT(*) as consumer_count,
    ROUND(AVG(monthly_spend_inr)::numeric, 0) as avg_spend,
    ROUND(AVG(daily_screen_time_hrs)::numeric, 1) as avg_screen_time
FROM chennai_consumers
GROUP BY age_group, primary_platform
ORDER BY age_group, consumer_count DESC;

-- Q2_purchase_by_genre
SELECT 
    top_content_genre,
    top_purchase_category,
    COUNT(*) as consumer_count,
    ROUND(AVG(monthly_spend_inr)::numeric, 0) as avg_monthly_spend,
    ROUND(AVG(daily_screen_time_hrs)::numeric, 2) as avg_screen_time
FROM chennai_consumers
GROUP BY top_content_genre, top_purchase_category
ORDER BY top_content_genre, consumer_count DESC;

-- Q3_locality_performance
SELECT 
    locality,
    COUNT(*) as store_count,
    ROUND(AVG(avg_monthly_sales_inr)::numeric, 0) as avg_sales,
    ROUND(AVG(avg_monthly_profit_inr)::numeric, 0) as avg_profit,
    ROUND(AVG(footfall_per_day)::numeric, 0) as avg_footfall,
    ROUND(AVG(rating)::numeric, 2) as avg_rating,
    SUM(CASE WHEN has_online_presence THEN 1 ELSE 0 END) 
        as online_stores
FROM chennai_stores
GROUP BY locality
ORDER BY avg_sales DESC;

-- Q4_sentiment_summary
SELECT 
    content_genre,
    COUNT(*) as total_posts,
    ROUND(AVG(sent_compound)::numeric, 3) as avg_sentiment,
    SUM(CASE WHEN sent_label = 'positive' THEN 1 ELSE 0 END) 
        as positive_count,
    SUM(CASE WHEN sent_label = 'negative' THEN 1 ELSE 0 END) 
        as negative_count,
    SUM(CASE WHEN sent_label = 'neutral'  THEN 1 ELSE 0 END) 
        as neutral_count,
    ROUND(AVG(views)::numeric, 0) as avg_views,
    ROUND(AVG(likes)::numeric, 0) as avg_likes
FROM sentiment_posts
GROUP BY content_genre
ORDER BY avg_sentiment DESC;

-- Q5_screentime_impulse
SELECT 
    impulse_buy_freq,
    COUNT(*) as consumer_count,
    ROUND(AVG(daily_screen_time_hrs)::numeric, 2) as avg_screen_time,
    ROUND(AVG(monthly_spend_inr)::numeric, 0) as avg_spend
FROM chennai_consumers
GROUP BY impulse_buy_freq
ORDER BY avg_screen_time DESC;

-- Q6_monthly_revenue
SELECT 
    month,
    category,
    total_revenue,
    festival_boost,
    is_trending,
    ROUND(
        (total_revenue - LAG(total_revenue) 
            OVER (PARTITION BY category ORDER BY month_date)
        )::numeric * 100.0 / NULLIF(
            LAG(total_revenue) 
            OVER (PARTITION BY category ORDER BY month_date), 0
        ), 1
    ) as month_over_month_pct
FROM monthly_trends
WHERE record_type = 'retail_sales'
ORDER BY category, month_date;

-- Q7_top_stores
SELECT 
    store_name,
    store_category,
    store_scale,
    locality,
    avg_monthly_sales_inr,
    avg_monthly_profit_inr,
    footfall_per_day,
    rating,
    has_online_presence,
    years_in_business
FROM chennai_stores
WHERE avg_monthly_sales_inr > (
    SELECT AVG(avg_monthly_sales_inr) 
    FROM chennai_stores
)
ORDER BY avg_monthly_sales_inr DESC
LIMIT 20;

-- Q8_consumer_segments
SELECT 
    age_group,
    income_bracket,
    COUNT(*) as segment_size,
    ROUND(AVG(monthly_spend_inr)::numeric, 0) as avg_spend,
    ROUND(AVG(daily_screen_time_hrs)::numeric, 1) as avg_screen_time,
    MODE() WITHIN GROUP (ORDER BY primary_platform) 
        as dominant_platform,
    MODE() WITHIN GROUP (ORDER BY top_content_genre) 
        as dominant_genre,
    MODE() WITHIN GROUP (ORDER BY top_purchase_category) 
        as dominant_purchase
FROM chennai_consumers
GROUP BY age_group, income_bracket
ORDER BY age_group, avg_spend DESC;

