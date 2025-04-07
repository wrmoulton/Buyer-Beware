# Buyer Beware: AI Firewall - Clustering & Dynamic Phrase Expansion Module

This module powers the Clustering and GPT-based Phrase Expansion component of the Buyer Beware AI Firewall, designed to detect and expand on coded language in suspected human trafficking posts.


## Disclaimer:

This project is part of a human trafficking detection research effort. As such, it uses datasets that contain explicit, coded, and potentially offensive or vulgar language to reflect real-world posts associated with illicit activity. This content is used solely for academic and research purposes in an effort to detect and combat trafficking-related language patterns. Viewer discretion is advised.


## Features
- Parses human-labeled posts and extracts rationale-based phrases

- Embeds phrases using OpenAI's text-embedding-ada-002

- Clusters embedded phrases with KMeans (cosine normalized)

- Visualizes clusters with t-SNE (color by cluster, marker by source)

- Expands phrases using GPT-4 Turbo with cosine similarity filtering

- Tracks GPT vs rationale sources in a persistent SQLite database

## Quick Start
1. Clone the repo and set up a virtual environment
```

python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt

```

2. Create a .env file with your OpenAI API key

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

```

3. Run the main pipeline

```
python main.py
```

This will: Initialize the database, Parse phrases from 2000_Rationaled_posts.json, Embed unembedded phrases, Cluster the phrases with KMeans, Visualize results with t-SNE. I highly recommend commenting out the clustering and GPT generation after the first time running. 

4. Expand a cluster with GPT

Edit main.py to call:

```

from src.phrase_expander import expand_cluster
expand_cluster(cluster_id=12, similarity_threshold=0.75)

```

## Database Schema
All phrases are stored in db/lexicon.db, in the lexicon_terms table with:

- phrase: the actual phrase

- embedding: 1536D OpenAI vector (comma-separated)

- cluster_id: assigned cluster

- source_post_id: original post or GPT tag

- source_type: 'rationale' or 'gpt'`

## Visualization

```
from src.visualize import plot_tsne
plot_tsne()
```
    
Generates a scatter plot of all embeddings:
Circle = rationale phrase
X = GPT-generated phrase
Color = cluster

## Author

Clustering & Dynamic Phrase Expansion lead: William Moulton
Part of UCF Senior Design: Buyer Beware - AI Firewall for Human Trafficking Detection
