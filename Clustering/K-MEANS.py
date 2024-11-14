import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

# Ensure the dataset contains the necessary columns
print(f"Columns in dataset: {df.columns.tolist()}")
if not all(col in df.columns for col in features):
    raise ValueError("Some required columns are missing from the dataset!")

# Standardize the data
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# PCA transformation for visualization (reduce to 2 components)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[features])

# Check if PCA transformation is correct
print("PCA Components shape:", pca_components.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Apply K-Means with custom parameters
n_clusters = 4  # Adjustable parameter for K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_clusters = kmeans.fit_predict(pca_components)

# Silhouette Score for K-Means
score = silhouette_score(pca_components, kmeans_clusters)
print(f"K-Means Silhouette Score (n_clusters={n_clusters}): {score:.3f}")

# Plot the K-Means clustering result
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_clusters, cmap='viridis', marker='o')
plt.title(f'K-Means Clustering (n_clusters={n_clusters})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
