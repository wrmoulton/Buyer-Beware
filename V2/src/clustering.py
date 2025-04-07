import numpy as np
from sklearn.cluster import DBSCAN
from src.database import get_connection
import ast
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def fetch_embeddings():
    """Load all embedded phrases from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM lexicon_terms WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    ids = []
    vectors = []

    for row in rows:
        id, embedding_str = row
        embedding = np.array([float(x) for x in embedding_str.split(",")])
        ids.append(id)
        vectors.append(embedding)

    return ids, np.array(vectors)

def run_kmeans_clustering(n_clusters=20):
    ids, vectors = fetch_embeddings()
    vectors = normalize(vectors, axis=1)  # ensures cosine behavior

    print(f" Running KMeans on {len(vectors)} embeddings...")
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(vectors)

    update_cluster_ids(ids, labels)
    print(f" Assigned {n_clusters} clusters.")

def update_cluster_ids(ids, labels):
    """Update the DB with new cluster labels."""
    conn = get_connection()
    cursor = conn.cursor()

    for id, label in zip(ids, labels):
        cursor.execute("UPDATE lexicon_terms SET cluster_id = ? WHERE id = ?", (int(label), id))

    conn.commit()
    conn.close()

    
def preview_clusters(limit=5):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT cluster_id FROM lexicon_terms WHERE cluster_id IS NOT NULL")
    clusters = [row[0] for row in cursor.fetchall()]

    for cluster_id in clusters:
        cursor.execute(
            "SELECT phrase FROM lexicon_terms WHERE cluster_id = ? LIMIT ?",
            (cluster_id, limit)
        )
        phrases = [row[0] for row in cursor.fetchall()]
        print(f"\n🧠 Cluster {cluster_id}:\n  - " + "\n  - ".join(phrases))

    conn.close()