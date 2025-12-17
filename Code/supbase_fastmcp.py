


"""
FastMCP Vector Search Tool with Supabase (Using pgvector)
"""
import os
import json
import boto3
from dotenv import load_dotenv
from supabase.client import Client, create_client
from fastmcp import FastMCP, Context  # <--- Import Context here
import logging
import sys
from typing import List, Dict, Any, Optional
import warnings

# --- Configuration & Setup ---

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("AWS_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")


# Create FastMCP server
mcp = FastMCP("Vector Search Server")

# --- Helper Functions ---

def create_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    return create_client(url, key)

def create_bedrock_client():
    region = os.environ.get("AWS_REGION_NAME", "us-east-1")
    return boto3.client(service_name='bedrock-runtime', region_name=region)

def get_embedding(text: str, bedrock_client) -> List[float]:
    payload = {"inputText": text}
    body = json.dumps(payload)
    response = bedrock_client.invoke_model(
        body=body,
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")

# --- MCP Tools ---

@mcp.tool()
async def search_gameloft_content(
    query: str,
    date_start: str,
    date_end: str,
    limit: int = 10,
        toolCallId: str = None,  # n8n sends this
    sessionId: str = None,   # n8n sends this
    action: str = None,      # n8n sends this
    chatInput: str = None    # n8n sends this

) -> List[Dict[str, Any]]:
    """
    Search Gameloft content with AI-determined date filtering.
    """
    try:
        # ctx.session_id and other metadata are available here if needed
        # if ctx:
        #     logger.info(f"Request from session: {ctx.session_id}")
            # If the client sends extra data, it might be in ctx.meta or similar depending on the transport
            # But primarily, adding `ctx: Context` stops FastMCP from crashing on extra args.

        supabase = create_supabase_client()
        bedrock = create_bedrock_client()
        
        logger.info(f"Searching for: {query}")
        query_embedding = get_embedding(query, bedrock)
        logger.info(f"Query embedding obtained, length: {len(query_embedding)}")
        
        response = supabase.rpc('search_content_by_date_range', {
            'query_embedding': query_embedding,
            'start_date': date_start,
            'end_date': date_end,
            'similarity_threshold': 0.2,
            'result_limit': limit
        }).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return [{"error": f"Search failed: {str(e)}"}]
    

@mcp.tool()
async def test_vector_connection(
    
        toolCallId: str = None,  # n8n sends this
    sessionId: str = None,   # n8n sends this
    action: str = None,      # n8n sends this
    chatInput: str = None    # n8n sends this
) -> Dict[str, Any]:
    """Test the vector search setup and return database info."""
    
    logger.info("=== Starting database connection test ===")
    
    # if ctx:
    #     logger.info(f"Session ID: {ctx.session_id}")

    try:
        supabase = create_supabase_client()
        logger.info("Supabase client created successfully")
        
        # 1. Test basic connection
        response = supabase.table("documents").select("count", count="exact").execute()
        document_count = response.count
        
        # 2. Test pgvector extension logic
        pgvector_available = False
        try:
            # FIX: We must pass dummy arguments so Supabase finds the function
            dummy_vector = [0.0] * 1024  # Match your Titan model dimension
            
            supabase.rpc('search_content_by_date_range', {
                'query_embedding': dummy_vector,
                'start_date': '2020-01-01',
                'end_date': '2020-01-02',
                'similarity_threshold': 0.0,
                'result_limit': 1
            }).execute()
            
            # If the above line runs without error, the function exists and works
            pgvector_available = True
            
        except Exception as e:
            logger.warning(f"Vector check warning: {e}")
            # Optional: Log the specific error to help debug
            if "dimension mismatch" in str(e):
                logger.error("Dimension mismatch! Check if SQL is 1024 and Python is 1024.")

        result = {
            "status": "success",
            "document_count": document_count,
            "vector_function_callable": pgvector_available
        }
        
        return result
        
    except Exception as e:
        logger.exception("Database connection test failed")
        return {
            "status": "error",
            "error": str(e)
        } 

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8002
        )