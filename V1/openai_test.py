import openai
import numpy as np
from numpy.linalg import norm
from database import LexiconDatabase
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Lexicon of terms
lexicon = [
    "69", "A-level", "ATM", "AR", "BBS", "BBBJ", "BBBJTC", "BBW", "B & D", "BDSM", "Bondage", 
    "BJ", "BJTC", "BLS", "BS", "CBJ", "CBT", "CD", "CIM", "CIMWS", "COF", "COB", "DATY", 
    "DATO", "DP", "DDP", "DFK", "DT", "Facial", "FE", "Filming", "Fire and ice", "Fisting", 
    "FK", "Foot fetish", "Foot job", "French", "FS", "Gagging", "GFE", "Greek", "GS", 
    "Happy ending", "HJ", "Italian", "LK", "MILF", "MFF", "MMF", "Mutual French", 
    "Mutual Natural French", "MSOG", "NSA", "OWO", "Pegging", "PSE", "R&T", "Rimming", 
    "Russian", "Snowballing", "Spanish", "Squirting", "Strap on", "Strip tease", 
    "Tea bagging", "Toy show", "Tromboning", "TTM", "Water sports"
]

# Function to get OpenAI embeddings (Updated for OpenAI API v1.0)
def get_embedding(text):
    """Fetch OpenAI embedding for a given text with error handling."""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error fetching embedding for '{text}': {e}")
        return None

# Initialize database and store lexicon terms
print("Initializing database and storing lexicon terms...")
db = LexiconDatabase()

# Store terms and their embeddings in the database
for term in lexicon:
    embedding = get_embedding(term)
    if embedding is not None:
        # Convert numpy array to bytes for storage
        embedding_bytes = embedding.tobytes()
        db.add_term(term, embedding_bytes)

print(f"Successfully stored {len(lexicon)} terms in the database.")

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Function to find the most similar terms in the lexicon
def find_similar_terms(new_term, top_n=3):
    """Finds the top N most similar terms in the lexicon for a new term."""
    new_embedding = get_embedding(new_term)
    
    if new_embedding is None:
        return []  # Return empty list if embedding fails

    # Get all terms and their embeddings from the database
    all_terms = db.get_all_terms()
    similarities = {}
    
    for term in all_terms:
        embedding_bytes = db.get_term_embedding(term)
        if embedding_bytes:
            stored_embedding = np.frombuffer(embedding_bytes)
            similarities[term] = cosine_similarity(new_embedding, stored_embedding)
    
    sorted_terms = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:top_n]  # Return top N most similar terms

# Function to generate related phrases using GPT-4
def generate_related_phrases(term):
    """Generates alternative phrases similar to the given term using GPT-4."""
    prompt = f"Generate 5 alternative phrases that mean the same as '{term}'."

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating phrases for '{term}': {e}")
        return "No alternative phrases generated."

# Test with a new term
new_term = "Around the World"
similar_terms = find_similar_terms(new_term)
related_phrases = generate_related_phrases(new_term)

# Print results
print("\nNew Term:", new_term)
print("Most Similar Terms:")
for term, score in similar_terms:
    print(f"  - {term} (Similarity: {score:.2f})")

print("\n🔹 Generated Related Phrases:")
print(related_phrases)

# Close the database connection
db.close()
