import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_Data2.csv')

# Define the features for clustering
features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
            'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
            'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
            'Satisfaction Score']

# Standardize the features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Perform PCA for dimensionality reduction (reduce to 2 components for visualization)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[features])

# Apply DBSCAN clustering with custom parameters
eps = 0.48  # Adjust the eps parameter (distance for neighbors)
min_samples = 3  # Adjust the min_samples (minimum points in a neighborhood)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_clusters = dbscan.fit_predict(pca_components)

# Check the number of clusters and noise points (-1)
unique_labels = set(dbscan_clusters)
print(f"Number of clusters: {len(unique_labels) - 1}")  # Excluding noise (-1)
print(f"Noise points: {dbscan_clusters.tolist().count(-1)}")

# Calculate silhouette score only if valid clusters are found (ignoring noise)
if len(unique_labels) > 1 and -1 not in unique_labels:
    score = silhouette_score(pca_components, dbscan_clusters)
    print(f"DBSCAN Silhouette Score (eps={eps}, min_samples={min_samples}): {score:.3f}")
else:
    print("DBSCAN did not find valid clusters.")

# Plot DBSCAN results if valid clusters are found
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan_clusters, cmap='viridis', marker='o')
plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
