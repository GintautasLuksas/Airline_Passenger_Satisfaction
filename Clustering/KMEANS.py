import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_Data2.csv')

    features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                'Satisfaction Score']

    print(f"Columns in dataset: {df.columns.tolist()}")
    if not all(col in df.columns for col in features):
        raise ValueError("Some required columns are missing from the dataset!")

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df[features])

    print("Explained Variance for each component:", pca.explained_variance_)
    print("Explained Variance Ratio for each component:", pca.explained_variance_ratio_)
    print("Cumulative Explained Variance Ratio:", pca.explained_variance_ratio_.cumsum())
    print("PCA Components shape:", pca_components.shape)

    option = input(
        "Choose an option:\n1. Manually input number of clusters\n2. Automatically scan for best number of clusters (1-15)\n")

    if option == '1':
        n_clusters = int(input("Enter the number of clusters for K-Means: "))
        plot_elbow_method(pca_components)
        plot_clusters(pca_components, n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_clusters = kmeans.fit_predict(pca_components)
        score = silhouette_score(pca_components, kmeans_clusters)
        print(f"K-Means Silhouette Score (n_clusters={n_clusters}): {score:.3f}")
    elif option == '2':
        scan_best_clusters(pca_components)
    else:
        print("Invalid choice. Please select option 1 or 2.")


def plot_elbow_method(pca_components):
    wcss = []
    for i in range(1, 16):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(pca_components)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 16), wcss, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(range(1, 16))
    plt.show()


def plot_clusters(pca_components, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(pca_components)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_clusters, cmap='viridis', marker='o')
    plt.title(f'K-Means Clustering (n_clusters={n_clusters})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()


def scan_best_clusters(pca_components):
    best_score = -1
    best_n_clusters = 0
    print("\nSilhouette Scores for different cluster numbers:")
    for n_clusters in range(2, 16):  # Starting from 2 clusters, as silhouette score needs at least 2 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_clusters = kmeans.fit_predict(pca_components)
        score = silhouette_score(pca_components, kmeans_clusters)
        print(f"Silhouette Score (n_clusters={n_clusters}): {score:.3f}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print(f"\nBest number of clusters based on silhouette score: {best_n_clusters} with score: {best_score:.3f}")
    plot_clusters(pca_components, best_n_clusters)


if __name__ == "__main__":
    main()
