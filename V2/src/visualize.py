import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from src.database import get_connection

def fetch_clustered_embeddings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT phrase, embedding, cluster_id FROM lexicon_terms WHERE embedding IS NOT NULL AND cluster_id IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    phrases = []
    vectors = []
    cluster_ids = []

    for phrase, embedding_str, cluster_id in rows:
        emb = np.array([float(x) for x in embedding_str.split(",")])
        phrases.append(phrase)
        vectors.append(emb)
        cluster_ids.append(cluster_id)

    return phrases, np.array(vectors), cluster_ids

def plot_tsne():
    phrases, vectors, cluster_ids = fetch_clustered_embeddings()
    vectors = normalize(vectors)

    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_ids, cmap='tab20', s=10)
    plt.title("t-SNE Visualization of Clustered Embeddings")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.show()
