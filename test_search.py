import sqlite3
import sqlite_vec
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search_products(query_text, top_k=5):
    # Get embedding for query
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Connect to database
    conn = sqlite3.connect('data/affiliate_links.db')
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    
    # Perform vector search
    results = conn.execute('''
        SELECT 
            a.name,
            a.description,
            a.category,
            a.click_url,
            vec_distance_cosine(v.embedding, ?) as distance
        FROM vec_links v
        JOIN affiliate_links a ON v.link_id = a.link_id
        ORDER BY distance
        LIMIT ?
    ''', (json.dumps(query_embedding), top_k)).fetchall()
    
    conn.close()
    return results

# Test it
print("Testing vector search for affiliate products...")
print("=" * 60)

# Test 1: Romantic gift search
print("\nQuery: 'romantic gift for anniversary'")
print("-" * 40)
results = search_products("romantic gift for anniversary")
for name, desc, cat, url, distance in results:
    print(f"• {name}")
    print(f"  Category: {cat}")
    print(f"  Similarity Score: {1-distance:.3f}")
    print(f"  Description: {desc[:100]}..." if desc and len(desc) > 100 else f"  Description: {desc}")
    print()

# Test 2: Tech product search
print("\nQuery: 'gaming laptop for students'")
print("-" * 40)
results = search_products("gaming laptop for students")
for name, desc, cat, url, distance in results:
    print(f"• {name}")
    print(f"  Category: {cat}")
    print(f"  Similarity Score: {1-distance:.3f}")
    print(f"  Description: {desc[:100]}..." if desc and len(desc) > 100 else f"  Description: {desc}")
    print()

# Test 3: Fashion search
print("\nQuery: 'summer dress for beach vacation'")
print("-" * 40)
results = search_products("summer dress for beach vacation")
for name, desc, cat, url, distance in results:
    print(f"• {name}")
    print(f"  Category: {cat}")
    print(f"  Similarity Score: {1-distance:.3f}")
    print(f"  Description: {desc[:100]}..." if desc and len(desc) > 100 else f"  Description: {desc}")
    print()

print("=" * 60)
print("Vector search test completed successfully!")