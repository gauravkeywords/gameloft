import requests
import trafilatura
import json
import logging
import sqlite3
import os
import time
import boto3
from datetime import datetime, date
from urllib.parse import urlparse
from trafilatura.settings import use_config
from dotenv import load_dotenv
from supabase.client import Client, create_client
from typing import List, Dict, Any

# LangChain imports
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_aws import ChatBedrock
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Configuration & Helper Functions ---

# Define a standard browser User-Agent to avoid 403 Bot blocks
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Create a Trafilatura configuration object
TRAF_CONFIG = use_config()
TRAF_CONFIG.set("DEFAULT", "USER_AGENT", BROWSER_UA)

# Embedding Configuration
EMBEDDING_MODEL_ID = os.environ.get("MODEL_ID_EMBEDDING") 
CHUNK_SIZE = 500 
CHUNK_OVERLAP = 100

def make_serializable(obj):
    """Convert datetime/date objects to ISO strings for JSON serialization."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj

def get_domain(url):
    """Extract domain name from URL."""
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return ""

def fetch_with_strategy(url):
    """
    Attempts to download URL content with a fallback strategy:
    1. Try standard download with Browser User-Agent.
    2. If that fails (likely SSL error), try disabling SSL verification.
    """
    try:
        # Attempt 1: Standard fetch with custom config
        downloaded = trafilatura.fetch_url(url, config=TRAF_CONFIG)
        
        # Attempt 2: If failed, try ignoring SSL cert errors (fixes SSLCertVerificationError)
        if downloaded is None:
            logger.warning(f"Standard fetch failed for {url}. Retrying with no_ssl=True...")
            downloaded = trafilatura.fetch_url(url, config=TRAF_CONFIG, no_ssl=True)
            
        return downloaded
    except Exception as e:
        logger.error(f"Critical fetch error for {url}: {e}")
        return None

# --- 2. SQLite Database Management ---

class NewsDatabase:
    def __init__(self, db_path="news_articles.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create the database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                published_date TEXT,
                source_name TEXT,
                img_src TEXT,
                thumbnail TEXT,
                searxng_data TEXT,
                processed INTEGER DEFAULT 0,
                trafilatura_success INTEGER DEFAULT 0,
                extracted_content TEXT,
                extracted_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index on URL for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON articles(url)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed ON articles(processed)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def insert_searxng_result(self, searx_article):
        """Insert a SearXNG result into the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            url = searx_article.get('url', '')
            if not url:
                return False
            
            # Extract data from SearXNG result
            title = searx_article.get('title', '')
            content = searx_article.get('content', '')
            published_date = searx_article.get('publishedDate', '')
            
            # Handle source - it might be a dict or string
            source = searx_article.get('source', {})
            source_name = source.get('name', '') if isinstance(source, dict) else str(source)
            
            img_src = searx_article.get('img_src', '')
            thumbnail = searx_article.get('thumbnail', '')
            searxng_data = json.dumps(searx_article)
            
            cursor.execute('''
                INSERT OR IGNORE INTO articles 
                (url, title, content, published_date, source_name, img_src, thumbnail, searxng_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (url, title, content, published_date, source_name, img_src, thumbnail, searxng_data))
            
            conn.commit()
            return cursor.rowcount > 0  # Returns True if new row was inserted
            
        except Exception as e:
            logger.error(f"Error inserting article: {e}")
            return False
        finally:
            conn.close()
    
    def get_unprocessed_articles(self, limit=None):
        """Get articles that haven't been processed yet."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM articles WHERE processed = 0 ORDER BY created_at'
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def mark_as_processed(self, url, success=True, extracted_content=None, extracted_metadata=None):
        """Mark an article as processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE articles 
                SET processed = 1, 
                    trafilatura_success = ?, 
                    extracted_content = ?, 
                    extracted_metadata = ?
                WHERE url = ?
            ''', (1 if success else 0, extracted_content, extracted_metadata, url))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error marking article as processed: {e}")
        finally:
            conn.close()
    
    def get_stats(self):
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM articles')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM articles WHERE processed = 1')
        processed = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM articles WHERE trafilatura_success = 1')
        successful = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total,
            'processed': processed,
            'unprocessed': total - processed,
            'trafilatura_successful': successful,
            'trafilatura_failed': processed - successful
        }

# --- 3. Search Component (SearXNG) ---

def get_searxng_news(query, base_url, time_range=None, page=1):
    """
    Search SearXNG news category.
    time_range options: hour, day, week, month, year
    """
    # Ensure URL ends with /search
    search_endpoint = f"{base_url.rstrip('/')}/search"
    
    params = {
        "q": query,
        "categories": "news",
        "format": "json",
        "language": "en-US",
        "pageno": page
    }
    if time_range:
        params["time_range"] = time_range

    try:
        logger.info(f"Searching SearXNG for: {query} (page {page})")
        response = requests.get(search_endpoint, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        logger.info(f"Found {len(results)} results from SearXNG page {page}.")
        return results

    except Exception as e:
        logger.error(f"Error querying SearXNG: {e}")
        return []

# --- 4. Extraction Component (Enhanced Trafilatura) ---

def extract_and_format_enhanced(db_article):
    """
    Enhanced extraction that uses database record and falls back to SearXNG content.
    """
    url = db_article.get('url')
    
    if not url:
        return None

    try:
        # A. Try Trafilatura Extraction
        downloaded = fetch_with_strategy(url)
        trafilatura_success = False
        trafilatura_content = None
        trafilatura_metadata = {}
        
        if downloaded:
            # Extract content and metadata with Trafilatura
            trafilatura_content = trafilatura.extract(
                downloaded, 
                include_comments=False, 
                favor_recall=True, 
                deduplicate=True,
                config=TRAF_CONFIG
            )
            
            traf_meta = trafilatura.extract_metadata(downloaded)
            trafilatura_metadata = traf_meta.as_dict() if traf_meta else {}
            
            if trafilatura_content:
                trafilatura_success = True

        # B. Determine final content (Trafilatura or SearXNG fallback)
        if trafilatura_success:
            final_content = trafilatura_content
            content_source = "trafilatura"
            logger.info(f"Using Trafilatura content for: {url}")
        else:
            # Fall back to SearXNG content
            final_content = db_article.get('content', '') or db_article.get('title', '')
            content_source = "searxng_fallback"
            logger.warning(f"Using SearXNG fallback content for: {url}")
            
            if not final_content:
                logger.error(f"No content available from either source for: {url}")
                return None

        # C. Construct comprehensive metadata
        clean_metadata = {
            "title": trafilatura_metadata.get('title') or db_article.get('title', ''),
            "date": trafilatura_metadata.get('date') or db_article.get('published_date', ''),
            "author": trafilatura_metadata.get('author', ''),
            "source": trafilatura_metadata.get('sitename') or db_article.get('source_name', '') or get_domain(url),
            "url": url,
            "description": (trafilatura_metadata.get('description') or db_article.get('content', ''))[:500],
            "image_url": trafilatura_metadata.get('image') or db_article.get('img_src', '') or db_article.get('thumbnail', ''),
            "content_source": content_source,
            "trafilatura_success": trafilatura_success
        }

        # D. Final output structure
        output = {
            "id": url,                    
            "page_content": final_content, 
            "metadata": clean_metadata    
        }
        
        # E. Update database with processing results
        db.mark_as_processed(
            url, 
            success=trafilatura_success,
            extracted_content=final_content if trafilatura_success else None,
            extracted_metadata=json.dumps(clean_metadata)
        )
        
        return output

    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        # Mark as processed but failed
        db.mark_as_processed(url, success=False)
        return None

# --- 5. Supabase Integration ---

class CustomBedrockEmbeddings(Embeddings):
    def __init__(self, bedrock_client, model_id: str = EMBEDDING_MODEL_ID):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self.get_embedding(text)
    
    def get_embedding(self, text: str) -> List[float]:
        payload = {"inputText": text}
        body = json.dumps(payload)
        response = self.bedrock_client.invoke_model(
            body=body,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")

def create_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    return create_client(url, key)

def create_bedrock_client():
    region = os.environ.get("AWS_REGION_NAME")
    return boto3.client(service_name='bedrock-runtime', region_name=region)

def process_and_upload_json_records(records: List[Dict[str, Any]]):
    """
    Loops over records, chunks them, generates embeddings manually, 
    and inserts raw data (ignoring ID) to Supabase.
    """
    if not records:
        logger.info("No records to upload to Supabase.")
        return
    
    # Initialize clients
    try:
        supabase: Client = create_supabase_client()
        bedrock = create_bedrock_client()
        embeddings = CustomBedrockEmbeddings(bedrock_client=bedrock)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase/Bedrock clients: {e}")
        return
    
    # Initialize splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    logger.info(f"Processing {len(records)} records for Supabase upload...")

    # Lists to hold batch data
    texts_to_embed = []
    metadatas = []

    for record in records:
        content = record.get("page_content", "")
        if not content or len(content.strip()) < 50:  # Skip very short content
            continue
            
        # Merge metadata
        combined_metadata = record.get("metadata", {}).copy()
        combined_metadata["source_id"] = record.get("id")

        # Split text
        chunks = text_splitter.split_text(content)

        # Collect chunks and metadata for batch processing
        for chunk in chunks:
            if len(chunk.strip()) > 20:  # Only include meaningful chunks
                texts_to_embed.append(chunk)
                metadatas.append(combined_metadata)

    if not texts_to_embed:
        logger.warning("No content to process for Supabase upload.")
        return

    try:
        logger.info(f"Generating embeddings for {len(texts_to_embed)} chunks...")
        
        # Generate Vectors
        vectors = embeddings.embed_documents(texts_to_embed)

        # Prepare the payload for Supabase
        data_to_insert = []
        for text, meta, vector in zip(texts_to_embed, metadatas, vectors):
            data_to_insert.append({
                "content": text,
                "metadata": meta,
                "embedding": vector
            })

        logger.info(f"Inserting {len(data_to_insert)} rows into Supabase...")
        
        # Use raw Supabase client to insert
        response = supabase.table("documents").insert(data_to_insert).execute()
        
        logger.info("Supabase upload complete!")
        
    except Exception as e:
        logger.error(f"Error uploading to Supabase: {e}")

# --- 6. Main Enhanced Pipeline ---

def collect_searxng_articles(topic, searx_url, max_pages=20, time_range=None):
    """
    Phase 1: Collect articles from SearXNG and store in SQLite database.
    """
    logger.info(f"=== PHASE 1: Collecting articles from SearXNG ===")
    logger.info(f"Topic: {topic}, Max Pages: {max_pages}")
    
    total_new_articles = 0
    
    for page_num in range(1, max_pages + 1):
        # Get items for current page
        items = get_searxng_news(topic, searx_url, time_range=time_range, page=page_num)
        
        if not items:
            logger.info(f"No more results found at page {page_num}. Stopping search.")
            break
        
        # Insert items into database
        new_items_count = 0
        for item in items:
            if db.insert_searxng_result(item):
                new_items_count += 1
        
        total_new_articles += new_items_count
        logger.info(f"Page {page_num}: Added {new_items_count} new unique articles.")
        
        # Polite delay
        time.sleep(1.5)
    
    # Show statistics
    stats = db.get_stats()
    logger.info(f"Collection complete! Added {total_new_articles} new articles.")
    logger.info(f"Database stats: {stats}")
    
    return total_new_articles

def process_stored_articles(limit=None):
    """
    Phase 2: Process stored articles with Trafilatura and prepare for Supabase.
    """
    logger.info(f"=== PHASE 2: Processing stored articles ===")
    
    # Get unprocessed articles
    articles = db.get_unprocessed_articles(limit=limit)
    
    if not articles:
        logger.info("No unprocessed articles found.")
        return []
    
    logger.info(f"Processing {len(articles)} unprocessed articles...")
    
    processed_records = []
    
    for i, article in enumerate(articles, 1):
        logger.info(f"Processing article {i}/{len(articles)}: {article['title'][:50]}...")
        
        result = extract_and_format_enhanced(article)
        if result:
            processed_records.append(result)
            logger.info(f"✓ Successfully processed: {article['url']}")
        else:
            logger.warning(f"✗ Failed to process: {article['url']}")
        
        # Small delay between processing
        time.sleep(0.5)
    
    logger.info(f"Processing complete! Successfully processed {len(processed_records)} articles.")
    
    return processed_records

def run_complete_pipeline(topic, searx_url, max_pages=20, time_range="week", process_limit=None):
    """
    Complete pipeline: Collect from SearXNG, process with Trafilatura, upload to Supabase.
    """
    logger.info(f"🚀 Starting Complete News Pipeline for: {topic}")
    
    # Phase 1: Collect articles
    new_articles = collect_searxng_articles(topic, searx_url, max_pages, time_range)
    
    if new_articles == 0:
        logger.info("No new articles collected. Checking for existing unprocessed articles...")
    
    # Phase 2: Process articles
    processed_records = process_stored_articles(limit=process_limit)
    
    if not processed_records:
        logger.info("No articles to upload to Supabase.")
        return
    
    # Phase 3: Upload to Supabase
    logger.info(f"=== PHASE 3: Uploading to Supabase ===")
    process_and_upload_json_records(processed_records)
    
    # Final statistics
    stats = db.get_stats()
    logger.info(f"🎉 Pipeline Complete!")
    logger.info(f"Final Database Stats: {stats}")
    logger.info(f"Processed Records for Supabase: {len(processed_records)}")

# --- 7. Global Database Instance ---
db = NewsDatabase()


if __name__ == "__main__":
    # --- Configuration ---
    SEARXNG_BASE_URL = "http://98.84.126.223:8080"

    # Your list of topics
    SEARCH_TOPICS = [
        # --- 🏢 Gameloft Corporate News ---
        "Gameloft official press release news",
        "Gameloft financial results and reports",
        "Gameloft new game announcements 2024 2025",
        "Gameloft Vivendi news updates",

        # --- 🏎️ Racing & Major Titles (High Priority) ---
        "Disney Speedstorm patch notes latest season",
        "Disney Speedstorm redeem codes list active",
        "Disney Speedstorm roadmap update",
        "Asphalt Legends Unite patch notes update",
        "Asphalt Legends Unite daily events schedule",
        "Asphalt 8: Airborne latest update news",
        "Asphalt Nitro Mobile Premium update",

        # --- ✨ Disney Dreamlight Valley (Very Active) ---
        "Disney Dreamlight Valley premium shop refresh info",
        "Disney Dreamlight Valley patch notes update",
        "Disney Dreamlight Valley star path rewards",
        "Disney Magic Kingdoms event schedule",

        # --- ⚔️ Strategy & RPG (Regular Updates) ---
        "March of Empires update patch notes",
        "War Planet Online: Global Conquest new events",
        "Dungeon Hunter 6 tier list update meta",
        "Dungeon Hunter 5 special events news",
        "Iron Blade: Medieval Legends RPG update",
        "Heroes of the Dark latest news",
        "AutoDefense Gameloft update",
        "Idle Siege game update news",

        # --- 🧩 Casual, Arcade & Apple Arcade ---
        "The Oregon Trail Apple Arcade update changelog",
        "LEGO Star Wars: Castaways events schedule",
        "Disney Getaway Blast update news",
        "My Little Pony: Mane Merge new events",
        "My Little Pony: Friendship is Magic social events",
        "Minion Rush special mission rewards",
        "Song Pop 2 latest music playlist update",
        "Ballistic Baseball Apple Arcade news",
        "Carmen Sandiego Gameloft game news",

        # --- 🐉 Simulation & Tycoon ---
        "Dragon Mania Legends weekly event calendar",
        "Dragon Mania Legends dotw (dragon of the week)",
        "Ice Age Adventures HD update",
        "Ice Age Village events",
        "Little Big City 2 news",

        # --- 🔫 Action & Legacy Titles ---
        "Modern Combat 5: Blackout eSports news",
        "Sniper Fury clan wars update",
        "Gangstar Vegas events schedule",
        "Gangstar New Orleans update news",
        "Six Guns: Gang Showdown news",
        "Brothers in Arms 3 events",
        "Blitz Brigade update status",
        "Zombiewood game news",

        # --- 📦 Collections ---
        "Gameloft Explorers Collection Bundle news"
    ]
    print(f"📋 Loaded {len(SEARCH_TOPICS)} topics to process.")

    # --- The Loop ---
    for i, topic in enumerate(SEARCH_TOPICS, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Processing Topic {i}/{len(SEARCH_TOPICS)}: {topic}")
            logger.info(f"{'='*60}")

            run_complete_pipeline(
                topic=topic,
                searx_url=SEARXNG_BASE_URL,
                max_pages=10,          # Reduced pages per topic since you have many topics
                time_range=None,    # Keep it relevant to recent news
                process_limit=None
            )

            # IMPORTANT: Polite delay between switching topics to prevent 
            # the Search Engine from blocking you for "bot-like behavior"
            logger.info("💤 Resting for 5 seconds before next topic...")
            time.sleep(5) 

        except Exception as e:
            # This ensures if "Disney Speedstorm" crashes, "Minion Rush" still runs
            logger.error(f"❌ CRITICAL ERROR processing topic '{topic}': {e}")
            continue

    # --- Final Stats ---
    stats = db.get_stats()
    print(f"\n🏁 Batch Processing Complete!")
    print(f"📊 Final Database Statistics:")
    print(f"   Total Articles Stored: {stats['total']}")
    print(f"   Successfully Processed: {stats['processed']}")
    print(f"   Trafilatura Success Rate: {stats['trafilatura_successful']}")

