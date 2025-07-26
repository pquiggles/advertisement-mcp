from typing import List, Dict, Optional
from mcp.server.fastmcp import FastMCP
import sqlite3
import sqlite_vec
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()

# Configure logging to stdout (now safe with HTTP transport!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to stdout
        logging.FileHandler('affiliate_server.log')  # Also save to file
    ]
)
logger = logging.getLogger('affiliate-mcp-server')

mcp = FastMCP("affiliate-products")
client = OpenAI()

# Log server startup
logger.info("Starting Affiliate MCP Server")

# Database connection
try:
    conn = sqlite3.connect('data/affiliate_links.db')
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    logger.info("Successfully connected to database")
except Exception as e:
    logger.error(f"Database connection error: {e}")
    raise

@mcp.tool()
async def search_products(
    query: str,
    num_results: int = 5,
    min_epc: Optional[float] = None,
    category: Optional[str] = None
) -> List[Dict]:
    """Search for affiliate products based on a text query
    
    Args:
        query: Search query text
        num_results: Number of results to return (default: 5)
        min_epc: Minimum earnings per click filter (optional)
        category: Filter by specific category (optional)
    """
    logger.info(f"Search request - Query: '{query}', Results: {num_results}, MinEPC: {min_epc}, Category: {category}")
    
    try:
        # Get embedding for query
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Build query with optional filters
        base_query = '''
            SELECT 
                a.name, a.description, a.category,
                a.click_url, a.coupon_code, a.epc_7day,
                vec_distance_cosine(v.embedding, ?) as distance
            FROM vec_links v
            JOIN affiliate_links a ON v.link_id = a.link_id
        '''
        
        conditions = []
        params = [json.dumps(query_embedding)]
        
        if min_epc is not None:
            conditions.append("CAST(a.epc_7day AS REAL) >= ?")
            params.append(min_epc)
        
        if category:
            conditions.append("a.category = ?")
            params.append(category)
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += " ORDER BY distance LIMIT ?"
        params.append(num_results)
        
        results = conn.execute(base_query, params).fetchall()
        
        logger.info(f"Found {len(results)} products for query '{query}'")
        
        formatted_results = [{
            'name': r[0],
            'description': r[1],
            'category': r[2],
            'url': r[3],
            'coupon': r[4] if r[4] else None,
            'epc': float(r[5].replace('$', '').replace(' USD', '')) if r[5] and isinstance(r[5], str) else (float(r[5]) if r[5] else 0.0),
            'relevance': round(1 - r[6], 3),
            'formatted_link': f"🔗 [View {r[0]}]({r[3]})",
            'coupon_display': f"🎟️ Use code: {r[4]}" if r[4] else None
        } for r in results]
        
        # Log top result for monitoring
        if formatted_results:
            logger.info(f"Top result: {formatted_results[0]['name']} (relevance: {formatted_results[0]['relevance']})")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in search_products: {e}")
        raise

@mcp.tool()
async def get_categories() -> List[Dict]:
    """Get all available product categories with counts"""
    logger.info("Fetching product categories")
    
    try:
        results = conn.execute('''
            SELECT DISTINCT category, COUNT(*) as count
            FROM affiliate_links
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
        ''').fetchall()
        
        categories = [{'category': r[0], 'product_count': r[1]} for r in results]
        logger.info(f"Found {len(categories)} categories")
        return categories
        
    except Exception as e:
        logger.error(f"Error in get_categories: {e}")
        raise

@mcp.tool()
async def get_top_products(
    category: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """Get top products by earnings potential
    
    Args:
        category: Filter by specific category (optional)
        limit: Number of products to return (default: 10)
    """
    logger.info(f"Fetching top products - Category: {category}, Limit: {limit}")
    
    try:
        if category:
            query = '''
                SELECT name, description, category, click_url, 
                       coupon_code, epc_7day
                FROM affiliate_links
                WHERE category = ?
                ORDER BY CAST(epc_7day AS REAL) DESC
                LIMIT ?
            '''
            params = (category, limit)
        else:
            query = '''
                SELECT name, description, category, click_url, 
                       coupon_code, epc_7day
                FROM affiliate_links
                ORDER BY CAST(epc_7day AS REAL) DESC
                LIMIT ?
            '''
            params = (limit,)
        
        results = conn.execute(query, params).fetchall()
        
        products = [{
            'name': r[0],
            'description': r[1],
            'category': r[2],
            'url': r[3],
            'coupon': r[4] if r[4] else None,
            'epc': float(r[5].replace('$', '').replace(' USD', '')) if r[5] and isinstance(r[5], str) else (float(r[5]) if r[5] else 0.0),
            'formatted_link': f"🔗 [View {r[0]}]({r[3]})",
            'coupon_display': f"🎟️ Use code: {r[4]}" if r[4] else None
        } for r in results]
        
        logger.info(f"Returned {len(products)} top products")
        return products
        
    except Exception as e:
        logger.error(f"Error in get_top_products: {e}")
        raise

@mcp.tool()
async def get_product_stats() -> Dict:
    """Get overall statistics about the product database"""
    logger.info("Generating product statistics")
    
    try:
        stats = {}
        
        # Total products
        stats['total_products'] = conn.execute(
            'SELECT COUNT(*) FROM affiliate_links'
        ).fetchone()[0]
        
        # Category breakdown
        stats['categories'] = conn.execute(
            'SELECT COUNT(DISTINCT category) FROM affiliate_links WHERE category IS NOT NULL'
        ).fetchone()[0]
        
        # Average EPC
        avg_epc = conn.execute(
            'SELECT AVG(CAST(epc_7day AS REAL)) FROM affiliate_links WHERE epc_7day IS NOT NULL'
        ).fetchone()[0]
        stats['average_epc'] = round(avg_epc, 2) if avg_epc else 0
        
        # Products with coupons
        stats['products_with_coupons'] = conn.execute(
            'SELECT COUNT(*) FROM affiliate_links WHERE coupon_code IS NOT NULL'
        ).fetchone()[0]
        
        # Top categories by count
        top_cats = conn.execute('''
            SELECT category, COUNT(*) as count 
            FROM affiliate_links 
            WHERE category IS NOT NULL 
            GROUP BY category 
            ORDER BY count DESC 
            LIMIT 5
        ''').fetchall()
        stats['top_categories'] = [{'category': c[0], 'count': c[1]} for c in top_cats]
        
        logger.info(f"Statistics: {stats['total_products']} products across {stats['categories']} categories")
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_product_stats: {e}")
        raise

# Note: Request logging is handled by FastMCP internally when log_level is set
# We can track usage through our tool function logs
# Database cleanup happens automatically when the process ends

if __name__ == "__main__":
    # Run with streamable-http transport
    # This allows multiple clients to connect and enables logging to stdout
    logger.info("Starting HTTP server on http://localhost:8000/mcp")
    mcp.run(transport="streamable-http")