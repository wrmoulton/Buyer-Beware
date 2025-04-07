import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from src.database import get_connection

def fetch_clustered_embeddings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT phrase, embedding, cluster_id, source_type
        FROM lexicon_terms
        WHERE embedding IS NOT NULL AND cluster_id IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    phrases = []
    vectors = []
    cluster_ids = []
    source_types = []

    for phrase, embedding_str, cluster_id, source_type in rows:
        emb = np.array([float(x) for x in embedding_str.split(",")])
        phrases.append(phrase)
        vectors.append(emb)
        cluster_ids.append(cluster_id)
        source_types.append(source_type)

    return phrases, np.array(vectors), cluster_ids, source_types

def plot_tsneGPT():
    phrases, vectors, cluster_ids, source_types = fetch_clustered_embeddings()
    vectors = normalize(vectors)

    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    for i in range(len(reduced)):
        x, y = reduced[i]
        cluster_color = cluster_ids[i]
        marker = "o" if source_types[i] == "rationale" else "x"
        plt.scatter(x, y, c=f"C{cluster_color % 10}", marker=marker, s=25, alpha=0.8)

    plt.title("t-SNE: GPT (x) vs Rationale (o)")
    plt.tight_layout()
    plt.show()
