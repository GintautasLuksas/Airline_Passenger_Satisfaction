import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_Data2.csv')

# Features for clustering
features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
            'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
            'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
            'Satisfaction Score']

# Standardize the data
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# PCA transformation for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[features])

# Apply Agglomerative Clustering with custom parameters
n_clusters = 4  # Adjustable parameter
agglo = AgglomerativeClustering(n_clusters=n_clusters)
agglo_clusters = agglo.fit_predict(pca_components)

# Silhouette Score for Agglomerative Clustering
score = silhouette_score(pca_components, agglo_clusters)
print(f"Agglomerative Clustering Silhouette Score (n_clusters={n_clusters}): {score:.3f}")

# Plot the Agglomerative Clustering
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=agglo_clusters, cmap='viridis', marker='o')
plt.title('Agglomerative Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
