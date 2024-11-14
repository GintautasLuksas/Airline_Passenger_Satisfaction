import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix
import numpy as np
from itertools import product

# Load the dataset
df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_Data2.csv')

# Features used for clustering
features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
            'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Satisfaction Score']

# Standardize the data
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Apply PCA for dimensionality reduction (to 2 components for visualization)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[features])


# --- DBSCAN Clustering with Hyperparameter Tuning ---
def dbscan_tuning(pca_components):
    best_score = -1
    best_eps = None
    best_min_samples = None
    best_clusters = None

    for eps, min_samples in product(np.linspace(0.1, 0.8, 5), range(3, 8)):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(pca_components)

        # Skip if clustering has noise or only one cluster
        if len(set(clusters)) > 1 and -1 not in set(clusters):
            try:
                score = silhouette_score(pca_components, clusters)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_clusters = clusters
            except ValueError:
                continue

    return best_eps, best_min_samples, best_clusters, best_score


# Find optimal DBSCAN parameters
best_eps, best_min_samples, dbscan_clusters, dbscan_score = dbscan_tuning(pca_components)
if dbscan_clusters is not None:
    print(f"Best DBSCAN - eps: {best_eps:.3f}, min_samples: {best_min_samples}, Silhouette Score: {dbscan_score:.3f}")
else:
    print("DBSCAN did not find valid clusters.")


# --- K-Means Clustering ---
def kmeans_tuning(pca_components):
    best_score = -1
    best_k = None
    best_clusters = None

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(pca_components)
        score = silhouette_score(pca_components, clusters)
        if score > best_score:
            best_score = score
            best_k = k
            best_clusters = clusters

    return best_k, best_clusters, best_score


# Find optimal K-Means parameters
best_k, kmeans_clusters, kmeans_score = kmeans_tuning(pca_components)
print(f"Best K-Means - n_clusters: {best_k}, Silhouette Score: {kmeans_score:.3f}")


# --- Dunn Index Calculation ---
def dunn_index(points, labels):
    unique_labels = set(labels)
    if len(unique_labels) < 2 or -1 in unique_labels:
        return np.nan

    inter_distances = []
    intra_distances = []

    for label in unique_labels:
        cluster_points = points[labels == label]
        intra_distances.append(np.max(distance_matrix(cluster_points, cluster_points)))

        for other_label in unique_labels:
            if label != other_label:
                other_points = points[labels == other_label]
                inter_distances.append(np.min(distance_matrix(cluster_points, other_points)))

    min_inter = np.min(inter_distances)
    max_intra = np.max(intra_distances)

    return min_inter / max_intra if max_intra != 0 else np.nan


# Compute Dunn Index for DBSCAN if valid
if dbscan_clusters is not None:
    dbscan_dunn = dunn_index(pca_components, dbscan_clusters)
    print(f"DBSCAN Dunn Index: {dbscan_dunn:.3f}")

# Compute Dunn Index for K-Means
kmeans_dunn = dunn_index(pca_components, kmeans_clusters)
print(f"K-Means Dunn Index: {kmeans_dunn:.3f}")

# --- Visualization ---
plt.figure(figsize=(14, 6))

if dbscan_clusters is not None:
    plt.subplot(1, 2, 1)
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan_clusters, cmap='viridis', marker='o')
    plt.title('DBSCAN Clustering')
    plt.colorbar(label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

plt.subplot(1, 2, 2)
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_clusters, cmap='viridis', marker='o')
plt.title('K-Means Clustering')
plt.colorbar(label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()
