import os
from openai import OpenAI
from dotenv import load_dotenv
from src.database import get_connection, insert_term
from src.embedder import get_embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_related_phrases(base_phrase: str, n: int = 5) -> list[str]:
    prompt = (
        f"Generate {n} alternate wordings, slang phrases, or coded equivalents for the phrase:\n"
        f"\"{base_phrase}\"\n\n"
        f"Respond with a numbered list of phrases only, no explanations."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        text = response.choices[0].message.content.strip()
        lines = text.split("\n")
        phrases = [line.split(". ", 1)[-1].strip() for line in lines if line.strip()]
        return phrases
    except Exception as e:
        print(f" GPT error: {e}")
        return []

def expand_cluster(cluster_id: int, similarity_threshold: float = 0.75):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT phrase, embedding FROM lexicon_terms WHERE cluster_id = ? AND embedding IS NOT NULL", (cluster_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f" No phrases found in cluster {cluster_id}")
        return

    # Compute cluster centroid
    embeddings = [np.array([float(x) for x in row[1].split(",")]) for row in rows]
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)

    total_added = 0

    for phrase, _ in rows:
        print(f"\n Expanding: {phrase}")
        new_phrases = generate_related_phrases(phrase)

        for new_phrase in new_phrases:
            new_embedding = get_embedding(new_phrase)
            if new_embedding is None:
                continue

            similarity = cosine_similarity(centroid, new_embedding.reshape(1, -1))[0][0]
            print(f"   {new_phrase} (similarity: {similarity:.2f})")

            if similarity >= similarity_threshold:
                insert_term(new_phrase, embedding=new_embedding, cluster_id=cluster_id, source_post_id="gpt_expanded", source_type="gpt")
                total_added += 1
            else:
                print("     Rejected: too dissimilar")

    print(f"\n Added {total_added} new phrases to cluster {cluster_id}")
