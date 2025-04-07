import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from src.database import get_connection

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """Fetch OpenAI embedding for a given text with error handling."""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f" Error fetching embedding for '{text}': {e}")
        return None

def embed_all_unembedded_terms():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, phrase FROM lexicon_terms WHERE embedding IS NULL")
    rows = cursor.fetchall()

    print(f" Embedding {len(rows)} terms...")

    for row in rows:
        id, phrase = row
        embedding = get_embedding(phrase)
        if embedding is not None:
            embedding_str = ",".join(map(str, embedding))
            cursor.execute(
                "UPDATE lexicon_terms SET embedding = ? WHERE id = ?",
                (embedding_str, id)
            )
            print(f" Embedded: {phrase}")

    conn.commit()
    conn.close()
