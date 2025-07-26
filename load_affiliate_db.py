import os
import sqlite3
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
import sqlite_vec

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embedding_text(row):
    """Combine relevant fields into a single text for embedding"""
    # Combine name, description, keywords, and category for rich semantic matching
    text_parts = [
        f"Product: {row['NAME']}",
        f"Description: {row['DESCRIPTION']}",
        f"Keywords: {row['KEYWORDS']}",
        f"Category: {row['CATEGORY']}",
        f"Type: {row['PROMOTION TYPE']}"
    ]
    # Filter out NaN values and join
    return " | ".join([part for part in text_parts if not pd.isna(part.split(": ")[1])])

def get_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings from OpenAI API"""
    # OpenAI can handle up to 2048 inputs per request
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

def setup_database():
    """Create and setup the SQLite database with vector extension"""
    # Connect to database
    conn = sqlite3.connect('data/affiliate_links.db')
    
    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    # Drop existing tables to ensure clean slate
    conn.execute('DROP TABLE IF EXISTS vec_links')
    conn.execute('DROP TABLE IF EXISTS affiliate_links')
    
    # Create the main table
    conn.execute('''
        CREATE TABLE affiliate_links (
            link_id INTEGER PRIMARY KEY,
            advertiser TEXT,
            name TEXT,
            description TEXT,
            keywords TEXT,
            category TEXT,
            promotion_type TEXT,
            epc_7day REAL,
            epc_3month REAL,
            click_url TEXT,
            coupon_code TEXT,
            embedding_text TEXT,
            embedding BLOB
        )
    ''')
    
    # Create a virtual table for vector search
    conn.execute('''
        CREATE VIRTUAL TABLE vec_links
        USING vec0(
            link_id INTEGER PRIMARY KEY,
            embedding float[1536]  -- text-embedding-3-small dimension
        )
    ''')
    
    conn.commit()
    return conn

def load_affiliate_data():
    """Load and process affiliate links"""
    print("Loading affiliate links CSV...")
    df = pd.read_csv('data/affiliate_links.csv')
    
    # Remove duplicates, keeping the first occurrence of each LINK ID
    print(f"Original rows: {len(df)}")
    df = df.drop_duplicates(subset=['LINK ID'], keep='first')
    print(f"Rows after removing duplicates: {len(df)}")
    
    # Create embedding text for each row
    print("Creating embedding texts...")
    df['embedding_text'] = df.apply(create_embedding_text, axis=1)
    
    # Process in batches to avoid API limits
    batch_size = 100
    all_embeddings = []
    
    print(f"Generating embeddings for {len(df)} products...")
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_texts = batch['embedding_text'].tolist()
        
        print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}...")
        embeddings = get_embeddings(batch_texts)
        all_embeddings.extend(embeddings)
    
    # Add embeddings to dataframe
    df['embedding'] = all_embeddings
    
    # Setup database
    conn = setup_database()
    
    # Insert data
    print("Inserting data into database...")
    for _, row in df.iterrows():
        # Insert into main table
        conn.execute('''
            INSERT OR REPLACE INTO affiliate_links 
            (link_id, advertiser, name, description, keywords, category, 
             promotion_type, epc_7day, epc_3month, click_url, coupon_code, 
             embedding_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['LINK ID'],
            row['ADVERTISER'],
            row['NAME'],
            row['DESCRIPTION'],
            row['KEYWORDS'],
            row['CATEGORY'],
            row['PROMOTION TYPE'],
            row['SEVEN DAY EPC'],
            row['THREE MONTH EPC'],
            row['CLICK URL'],
            row['COUPON CODE'],
            row['embedding_text'],
            json.dumps(row['embedding'])  # Store as JSON blob
        ))
        
        # Insert into vector table
        conn.execute('''
            INSERT OR REPLACE INTO vec_links (link_id, embedding)
            VALUES (?, ?)
        ''', (row['LINK ID'], json.dumps(row['embedding'])))
    
    conn.commit()
    print(f"Successfully loaded {len(df)} affiliate links into the database!")
    
    # Create indexes for better performance
    conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON affiliate_links(category)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_epc ON affiliate_links(epc_7day DESC)')
    conn.commit()
    
    conn.close()

if __name__ == "__main__":
    load_affiliate_data()