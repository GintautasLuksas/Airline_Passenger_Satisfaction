import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

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

# Apply DBSCAN with custom parameters
eps = 0.1  # Adjustable parameter
min_samples = 5  # Adjustable parameter
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_clusters = dbscan.fit_predict(pca_components)

# Silhouette Score for DBSCAN
if len(set(dbscan_clusters)) > 1 and -1 not in set(dbscan_clusters):
    score = silhouette_score(pca_components, dbscan_clusters)
    print(f"DBSCAN Silhouette Score (eps={eps}, min_samples={min_samples}): {score:.3f}")
else:
    print("DBSCAN clustering did not produce valid clusters for Silhouette Score calculation.")

# Plot the DBSCAN clustering
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan_clusters, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
