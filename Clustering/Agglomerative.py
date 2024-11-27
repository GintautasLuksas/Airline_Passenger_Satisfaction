import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
import tkinter as tk
from tkinter import simpledialog

def dunn_index(X, labels):
    """
    Calculate the Dunn Index, which is the ratio of the minimum inter-cluster distance
    to the maximum intra-cluster distance.
    """
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return np.nan
    intra_cluster_distances = []
    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        if len(cluster_points) > 1:
            distances = pdist(cluster_points)
            intra_cluster_distances.append(np.max(distances))
        else:
            intra_cluster_distances.append(0)
    max_intra_distance = max(intra_cluster_distances)
    inter_cluster_distances = []
    for i, cluster1 in enumerate(unique_clusters[:-1]):
        for cluster2 in unique_clusters[i + 1:]:
            cluster1_points = X[labels == cluster1]
            cluster2_points = X[labels == cluster2]
            distances = cdist(cluster1_points, cluster2_points)
            inter_cluster_distances.append(np.min(distances))
    min_inter_distance = min(inter_cluster_distances)
    dunn = min_inter_distance / max_intra_distance if max_intra_distance > 0 else np.nan
    return dunn

def perform_agglomerative_clustering(df, n_clusters):
    """
    Perform Agglomerative Clustering with different linkage methods and visualize
    the results with scatter plots, heatmaps, and dendrograms.
    """
    features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                'Satisfaction Score']

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df[features])

    linkage_methods = ['ward', 'complete', 'average', 'single']
    method_scores = {}
    best_method = None
    best_score = -float('inf')

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    ax_idx = 0

    for linkage_method in linkage_methods:
        # Agglomerative clustering
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        agglo_clusters = agglo.fit_predict(pca_components)

        silhouette = silhouette_score(pca_components, agglo_clusters)
        print(f"Silhouette Score (n_clusters={n_clusters}, linkage='{linkage_method}'): {silhouette:.3f}")

        davies_bouldin = davies_bouldin_score(pca_components, agglo_clusters)
        print(f"Davies-Bouldin Index (n_clusters={n_clusters}, linkage='{linkage_method}'): {davies_bouldin:.3f}")

        dunn_score = dunn_index(pca_components, agglo_clusters)
        print(f"Dunn Index (n_clusters={n_clusters}, linkage='{linkage_method}'): {dunn_score:.3f}")

        method_scores[linkage_method] = silhouette

        scatter_ax = axes[ax_idx, 0]
        heatmap_ax = axes[ax_idx, 1]

        scatter_ax.scatter(pca_components[:, 0], pca_components[:, 1], c=agglo_clusters, cmap='viridis', marker='o')
        scatter_ax.set_title(f'Agglomerative Clustering ({linkage_method})', fontsize=16)
        scatter_ax.set_xlabel('PCA Component 1', fontsize=12)
        scatter_ax.set_ylabel('PCA Component 2', fontsize=12)

        cluster_counts = pd.Series(agglo_clusters).value_counts().sort_index()
        sns.heatmap(cluster_counts.values.reshape(-1, 1), annot=True, fmt='d', cmap='viridis', cbar=False,
                    ax=heatmap_ax)
        heatmap_ax.set_title(f'Cluster Distribution Heatmap ({linkage_method})', fontsize=16)
        heatmap_ax.set_xlabel('Clusters', fontsize=12)
        heatmap_ax.set_ylabel('Count', fontsize=12)

        ax_idx += 1

    best_method = max(method_scores, key=method_scores.get)
    print(f"\nBest method based on Silhouette Score: {best_method}")

    fig.suptitle(f'Agglomerative Clustering Results (n_clusters={n_clusters})', fontsize=18)

    for ax in axes.flatten():
        ax.set_xticks(np.arange(n_clusters))
        ax.set_xticklabels(np.arange(n_clusters))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0.043, 0.082, 0.92, 0.86])  # Adjusted layout
    plt.subplots_adjust(hspace=0.825, wspace=0.171)  # Adjusted subplot spacing

    # Plot dendrogram separately for better visualization
    linkage_matrix = linkage(pca_components, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title(f'Dendrogram (Ward linkage, n_clusters={n_clusters})', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.show()

def main():
    """
    Main function to load data, display a UI for selecting the number of clusters, and plot clustering comparisons.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    try:
        # Get the number of clusters from the user
        n_clusters = simpledialog.askinteger("Input", "Enter the number of clusters:", minvalue=2)
        if n_clusters:
            df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Cleaned_Data.csv')
            perform_agglomerative_clustering(df, n_clusters)
        else:
            print("No valid input provided.")
    except ValueError:
        print("Invalid input. Please enter a valid number of clusters.")

if __name__ == "__main__":
    main()
