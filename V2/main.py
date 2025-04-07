from src.database import initialize_database
from src.rationales_parser import parse_and_store_from_file
from src.embedder import embed_all_unembedded_terms
#from src.clustering import run_dbscan_clustering
from src.clustering import run_kmeans_clustering
from src.visualize import plot_tsne 
from src.clustering import preview_clusters
from src.database import add_source_type_column
from src.phrase_expander import expand_cluster
from src.visualizeGPT import plot_tsneGPT

if __name__ == "__main__":

    initialize_database() ##Creates db

    add_source_type_column() ##Added for GPT generated data label

    parse_and_store_from_file("data/2000_Rationaled_posts.json") ## parses rationlized data

    embed_all_unembedded_terms() ##Will Embed all terms

    #run_kmeans_clustering(n_clusters=25) ##Only need to run once for clustering 

    preview_clusters()

    plot_tsne() ##Plots clusters

    #expand_cluster(cluster_id=12, similarity_threshold=0.75) ##Only run when you want to expand clusters

    plot_tsneGPT() ##Plots clusters and shows which ones are GPT-generated and which are regular
