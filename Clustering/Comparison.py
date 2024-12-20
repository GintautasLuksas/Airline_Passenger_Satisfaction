import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import simpledialog
import numpy as np
from scipy.spatial.distance import pdist, cdist

def dunn_index(X, labels):
    """
    Calculate the Dunn Index, which is the ratio of the minimum inter-cluster distance
    to the maximum intra-cluster distance.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    labels : array-like, shape (n_samples,)
        Cluster labels.

    Returns:
    float
        The Dunn Index score.
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

def davies_bouldin_index(X, labels):
    """
    Calculate the Davies-Bouldin Index, a measure of clustering quality that evaluates
    the average similarity ratio of each cluster with the one most similar to it.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    labels : array-like, shape (n_samples,)
        Cluster labels.

    Returns:
    float
        The Davies-Bouldin Index score.
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        return np.nan
    distances = cdist(X, X)
    cluster_centers = [np.mean(X[labels == cluster], axis=0) for cluster in unique_clusters]
    db_index = 0
    for i in range(n_clusters):
        max_ratio = -np.inf
        for j in range(n_clusters):
            if i != j:
                si = np.mean(distances[labels == unique_clusters[i], :][:, labels == unique_clusters[i]])
                sj = np.mean(distances[labels == unique_clusters[j], :][:, labels == unique_clusters[j]])
                dij = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                ratio = (si + sj) / dij
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    return db_index / n_clusters

def plot_comparison(df, n_clusters):
    """
    Perform K-Means and Agglomerative Clustering, plot results side-by-side, and display silhouette, Dunn, and Davies-Bouldin scores.

    Parameters:
    df : DataFrame
        The input dataset with features for clustering.
    n_clusters : int
        The number of clusters for the clustering algorithms.

    Returns:
    None
    """
    features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                'Satisfaction Score']

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(pca_components)
    kmeans_score = silhouette_score(pca_components, kmeans_clusters)
    kmeans_dunn = dunn_index(pca_components, kmeans_clusters)
    kmeans_davies_bouldin = davies_bouldin_index(pca_components, kmeans_clusters)

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglomerative_clusters = agglomerative.fit_predict(pca_components)
    agglomerative_score = silhouette_score(pca_components, agglomerative_clusters)
    agglomerative_dunn = dunn_index(pca_components, agglomerative_clusters)
    agglomerative_davies_bouldin = davies_bouldin_index(pca_components, agglomerative_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_clusters, cmap='viridis', marker='o')
    axes[0].set_title(f'K-Means Clustering (n_clusters={n_clusters})\nSilhouette: {kmeans_score:.3f}, Dunn: {kmeans_dunn:.3f}, DB: {kmeans_davies_bouldin:.3f}', fontsize=14)
    axes[0].set_xlabel('PCA Component 1', fontsize=12)
    axes[0].set_ylabel('PCA Component 2', fontsize=12)

    axes[1].scatter(pca_components[:, 0], pca_components[:, 1], c=agglomerative_clusters, cmap='viridis', marker='o')
    axes[1].set_title(f'Agglomerative Clustering (n_clusters={n_clusters})\nSilhouette: {agglomerative_score:.3f}, Dunn: {agglomerative_dunn:.3f}, DB: {agglomerative_davies_bouldin:.3f}', fontsize=14)
    axes[1].set_xlabel('PCA Component 1', fontsize=12)
    axes[1].set_ylabel('PCA Component 2', fontsize=12)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load data, display a UI for selecting the number of clusters, and plot clustering comparisons.

    Returns:
    None
    """
    df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_Data2.csv')

    root = tk.Tk()
    root.withdraw()  # Hides the main tkinter window

    n_clusters = simpledialog.askinteger("Input", "Enter the number of clusters:", minvalue=2)
    if n_clusters:
        plot_comparison(df, n_clusters)
    else:
        print("No valid input provided.")

if __name__ == "__main__":
    main()
