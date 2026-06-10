import chromadb
import os
import sqlite3
from datetime import datetime
from chromadb.utils import embedding_functions

DB_PATH = "data/journal.db"
CHROMA_PATH = "data/chroma"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="journal_entries",
    embedding_function=embedding_fn
)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT,
            dominant_emotion TEXT,
            compound_score REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_entry(text, sentiment, dominant_emotion, compound_score, user_id=0):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO entries (text,sentiment,dominant_emotion,compound_score,timestamp,user_id)"
        " VALUES (?,?,?,?,?,?)",
        (text, sentiment, dominant_emotion, compound_score, timestamp, user_id)
    )
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    collection.add(
        documents=[text], ids=[str(entry_id)],
        metadatas=[{"timestamp": timestamp, "sentiment": sentiment,
                    "user_id": str(user_id)}]
    )
    return entry_id

def get_relevant_memories(text, n=3, user_id=0):
    try:
        # Filter by user_id in metadata
        where = {"user_id": str(user_id)} if user_id else None
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[text],
            n_results=min(n, count),
            where=where if where else None
        )
        if results and results['documents'][0]:
            memories = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                memories.append(f"[{meta['timestamp']}] {doc}")
            return memories
        return []
    except:
        return []

def get_all_entries(user_id=0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if user_id:
        cursor.execute(
            'SELECT * FROM entries WHERE user_id = ? ORDER BY timestamp DESC',
            (user_id,)
        )
    else:
        cursor.execute('SELECT * FROM entries ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()