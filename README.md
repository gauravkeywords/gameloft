# Gameloft Project Documentation

This document outlines the setup and installation of the various services for this project.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   **Docker**: 
*   **Docker Compose**:
*   **Miniconda**:

## Service Startup Order

For the services to work correctly, they should be started in the following order:

1.  **Supabase**
2.  **N8N**
3.  **Searxng**
4.  **OpenWebUI**

This ensures that all dependencies are met before each service starts.

## N8N Setup

To start the N8N service, please follow these steps:

1.  Navigate to the N8N project directory:
    ```bash
    cd /home/ubuntu/projects/N8N
    ```

2.  Start the service using Docker Compose:
    ```bash
    docker-compose up -d
    ```

This will start the N8N container in the background. You should be able to access the N8N web interface on port 5678.

## OpenWebUI Setup

To start the OpenWebUI service, please follow these steps:

1.  Navigate to the OpenWebUI project directory:
    ```bash
    cd /home/ubuntu/projects/openwebui
    ```

2.  Start the service using Docker Compose:
    ```bash
    docker-compose up -d
    ```

This will start the OpenWebUI container in the background. You should be able to access the OpenWebUI web interface on port 3000.

### N8N Pipe Function

To integrate OpenWebUI with N8N, you can use the following pipe function. This function allows you to send data from OpenWebUI to an N8N workflow.

1.  **Navigate to the Admin Settings**: In the OpenWebUI, go to the "Admin" section.
2.  **Create a New Pipe**: In the "Pipes" section, click "Create a new pipe".
3.  **Add the following code**:
        You can get pipe code from gameloft pepe function that we have created.
   
4.  **Configure the Valves**: You will need to set the following environment variables in your OpenWebUI environment:

    *   `N8N_BEARER_TOKEN`: Token
    *   `N8N_URL`: url
    *   `N8N_API_KEY`: `Api Key`
    *   `N8N_API_BASE_URL`: `Base URL`


## Searxng Setup

To start the Searxng service, please follow these steps:

1.  Navigate to the Searxng project directory:
    ```bash
    cd /home/ubuntu/projects/searxng
    ```

2.  Start the service using Docker Compose:
    ```bash
    docker-compose up -d
    ```

This will start the Searxng container in the background. You should be able to access the Searxng web interface on port 8080.

## Supabase Setup

To start the Supabase service, please follow these steps:

1.  Navigate to the Supabase project directory:
    ```bash
    cd /home/ubuntu/projects/supabase-project
    ```

2.  Start the service using Docker Compose:
    ```bash
    docker-compose up -d
    ```

This will start the Supabase container in the background. You should be able to access the Supabase API on port 8000.

## Gameloft Scraping Script

This project includes a Python script that scrapes news articles from various sources, processes the text, and stores the data in a local SQLite database and a Supabase instance.

### How it Works

The script (`scrap_gameloft.py`) performs the following actions:

1.  **Fetches URLs**: It retrieves URLs from various sources.
2.  **Scrapes Content**: It uses the `trafilatura` library to extract the main content and metadata from each URL.
3.  **Stores in SQLite**: The scraped data is stored in a local SQLite database (`news_articles.db`).
4.  **Generates Embeddings**: It uses AWS Bedrock to generate text embeddings for the article content.
5.  **Stores in Supabase**: The embeddings and other article data are stored in a Supabase database, which is used as a vector store for similarity searches.

### Credentials

To run the script, you will need to have a `.env` file in the `/home/ubuntu/projects/Code` directory with the following credentials:

*   **Supabase**:
    *   `SUPABASE_URL`: The URL of your Supabase instance.
    *   `SUPABASE_SERVICE_KEY`: Your Supabase service key.
*   **AWS**:
    *   `AWS_REGION_NAME`: The AWS region for the Bedrock service.
    *   `AWS_ACCESS_KEY_ID`: Your AWS access key ID.
    *   `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key.
*   **Embedding Model**:
    *   `MODEL_ID_EMBEDDING`: The ID of the embedding model to use (e.g., `amazon.titan-embed-text-v2:0`).

### How to Run Manually

1.  Navigate to the Code project directory:
    ```bash
    cd /home/ubuntu/projects/Code
    ```

2.  Activate the Conda environment:
    ```bash
    conda activate base
    ```

3.  Run the script:
    ```bash
    python scrap_gameloft.py
    ```

### Automated Execution (Cron Job)

The scraping script is configured to run automatically every day at 23:30 (11:30 PM) via a cron job. The cron job executes the `/home/ubuntu/projects/Code/start.sh` script, which handles the setup and execution of the Python script. The output of the script is logged to `/home/ubuntu/projects/Code/output.log`.

### Supabase Vector Search (FastMCP)

The `supbase_fastmcp.py` script implements a FastMCP server that provides a vector search tool for the Supabase database. This allows for semantic searching of the scraped news articles based on their content.

#### How it Works

1.  **FastMCP Server**: The script creates a FastMCP server that listens for requests on port 8002.
2.  **Supabase Integration**: It connects to the Supabase instance to perform vector searches using the `pgvector` extension.
3.  **AWS Bedrock**: It uses AWS Bedrock to generate text embeddings for the search queries.
4.  **SQL Function**: The script is designed to work with a specific SQL function (`search_content_by_date_range`) that must be created in the Supabase database. This function performs the similarity search and boosts the scores of recent articles.

    You can create the function by running the following SQL command in your Supabase SQL editor:

    ```sql
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
    ```

#### How to Run

1.  Navigate to the Code project directory:
    ```bash
    cd /home/ubuntu/projects/Code
    ```
2.  Activate the Conda environment:
    ```bash
    conda activate base
    ```

3.  Run the script:
    ```bash
    python supbase_fastmcp.py
    ```

## System Architecture

This project is a multi-service system designed to provide a chat-based interface for searching and retrieving news articles. The services work together as follows:

1.  **OpenWebUI (Frontend)**: This is the user-facing application. It provides a chat interface where users can enter queries. The OpenWebUI is configured with a pipe function that sends user messages to the N8N workflow.

2.  **N8N (Workflow Automation)**: N8N acts as the central hub for the system. It receives queries from OpenWebUI and orchestrates the workflow to retrieve the necessary information. It is responsible for calling the other services  the FastMCP server and returning the final result to the user.

3.  **Searxng (Web Search)**: When a query requires up-to-date information from the internet, N8N calls the Searxng service to perform a web search as well.

4.  **Supabase (Vector Database)**: Supabase is used as a vector database to store embeddings of the scraped news articles. This allows for semantic searching of the articles based on their content.

5.  **Gameloft Scraping Script (Data Ingestion)**: This Python script runs on a daily cron job to scrape news articles from various sources. It generates embeddings for the article content using AWS Bedrock and stores them in the Supabase database.

6.  **Supabase Vector Search (FastMCP)**: This Python script runs a FastMCP server that provides a simple API for searching the Supabase vector database. N8N calls this server to perform semantic searches on the scraped news articles.

In summary, the user interacts with the system through the OpenWebUI chat interface. The query is sent to N8N, which then uses Searxng for web searches and the FastMCP server to search the Supabase vector database. The results are then returned to the user through the OpenWebUI interface.
