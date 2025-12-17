CREATE OR REPLACE FUNCTION search_content_by_date_range(
    query_embedding vector(1024),
    start_date date,
    end_date date,
    similarity_threshold float DEFAULT 0.6,
    result_limit int DEFAULT 10
)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    similarity float,
    content_date date
) 
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity,
        (d.metadata->>'date')::date as content_date
    FROM documents d
    WHERE 
        (d.metadata->>'date')::date BETWEEN start_date AND end_date
        AND 1 - (d.embedding <=> query_embedding) > similarity_threshold
    ORDER BY 
        CASE 
            -- Boost very recent content (last 7 days from end_date)
            WHEN (d.metadata->>'date')::date >= end_date - INTERVAL '7 days' 
            THEN (1 - (d.embedding <=> query_embedding)) * 1.3
            -- Boost recent content (last 30 days from end_date)  
            WHEN (d.metadata->>'date')::date >= end_date - INTERVAL '30 days'
            THEN (1 - (d.embedding <=> query_embedding)) * 1.1
            ELSE 1 - (d.embedding <=> query_embedding)
        END DESC
    LIMIT result_limit;
END;
$$;