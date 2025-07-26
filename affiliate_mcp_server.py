from typing import List, Dict, Optional
from mcp.server.fastmcp import FastMCP
import sqlite3
import sqlite_vec
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("affiliate-products")
client = OpenAI()

# Database connection
conn = sqlite3.connect('data/affiliate_links.db')
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

@mcp.tool()
async def search_products(
    query: str,
    num_results: int = 5
) -> List[Dict]:
    """Search for affiliate products based on a text query"""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    results = conn.execute('''
        SELECT 
            a.name, a.description, a.category,
            a.click_url, a.coupon_code, a.epc_7day,
            vec_distance_cosine(v.embedding, ?) as distance
        FROM vec_links v
        JOIN affiliate_links a ON v.link_id = a.link_id
        ORDER BY distance
        LIMIT ?
    ''', (json.dumps(query_embedding), num_results)).fetchall()
    
    return [{
        'name': r[0],
        'description': r[1],
        'category': r[2],
        'url': r[3],
        'coupon': r[4] if r[4] else None,
        'epc': float(r[5].replace('$', '').replace(' USD', '')) if r[5] and isinstance(r[5], str) else (float(r[5]) if r[5] else 0.0),
        'relevance': round(1 - r[6], 3)
    } for r in results]

@mcp.tool()
async def get_categories() -> List[Dict]:
    """Get all available product categories"""
    results = conn.execute('''
        SELECT DISTINCT category, COUNT(*) as count
        FROM affiliate_links
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    ''').fetchall()
    
    return [{'category': r[0], 'product_count': r[1]} for r in results]

@mcp.tool()
async def get_top_products(
    category: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """Get top products by earnings potential"""
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
    
    return [{
        'name': r[0],
        'description': r[1],
        'category': r[2],
        'url': r[3],
        'coupon': r[4] if r[4] else None,
        'epc': float(r[5].replace('$', '').replace(' USD', '')) if r[5] and isinstance(r[5], str) else (float(r[5]) if r[5] else 0.0)
    } for r in results]

if __name__ == "__main__":
    mcp.run(transport='stdio')