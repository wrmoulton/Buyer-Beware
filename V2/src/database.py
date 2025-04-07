import sqlite3
import os

DB_PATH = os.path.join("db", "lexicon.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    os.makedirs("db", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def initialize_database():
    """Creates the lexicon_terms table if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lexicon_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phrase TEXT NOT NULL,
            embedding TEXT,  -- Stored as comma-separated string for simplicity
            cluster_id INTEGER,
            source_post_id TEXT
        )
    ''')

    conn.commit()
    conn.close()

def insert_term(phrase, embedding=None, cluster_id=None, source_post_id=None, source_type="rationale"):
    conn = get_connection()
    cursor = conn.cursor()

    # Skip duplicates from same source
    cursor.execute(
        "SELECT id FROM lexicon_terms WHERE phrase = ? AND source_post_id = ?",
        (phrase, source_post_id)
    )
    if cursor.fetchone():
        conn.close()
        return

    embedding_str = ",".join(map(str, embedding)) if embedding is not None else None

    cursor.execute('''
        INSERT INTO lexicon_terms (phrase, embedding, cluster_id, source_post_id, source_type)
        VALUES (?, ?, ?, ?, ?)
    ''', (phrase, embedding_str, cluster_id, source_post_id, source_type))

    conn.commit()
    conn.close()

def get_all_terms():
    """Fetch all phrases and their embeddings."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, phrase, embedding FROM lexicon_terms")
    results = cursor.fetchall()

    conn.close()
    return results

def clear_database():
    """Utility: clears the table for testing or re-runs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM lexicon_terms")
    conn.commit()
    conn.close()

def add_source_type_column():
    conn = get_connection()
    cursor = conn.cursor()

    # Only add if not already present
    cursor.execute("PRAGMA table_info(lexicon_terms)")
    columns = [row[1] for row in cursor.fetchall()]
    if "source_type" not in columns:
        cursor.execute("ALTER TABLE lexicon_terms ADD COLUMN source_type TEXT")
        conn.commit()
        print(" Added 'source_type' column to lexicon_terms.")
    else:
        print(" 'source_type' column already exists.")

    conn.close()
