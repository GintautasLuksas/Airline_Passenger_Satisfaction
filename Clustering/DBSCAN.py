import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from itertools import combinations, product


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def standardize_data(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def perform_pca(df, features, n_components=2):
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df[features])
    return pca_components


def dbscan_tuning(pca_components, feature_subset):
    best_score = -1
    best_eps = None
    best_min_samples = None
    best_clusters = None

    for eps, min_samples in product(np.linspace(0.1, 1.5, 10), range(3, 10)):  # Adjust eps and min_samples range
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(pca_components)
        unique_labels = set(clusters)

        if len(unique_labels) > 1 and -1 not in unique_labels:
            try:
                score = silhouette_score(pca_components, clusters)
                print(
                    f"Testing subset {feature_subset} | eps={eps:.3f}, min_samples={min_samples} | Score: {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_clusters = clusters
            except ValueError:
                continue

    return best_eps, best_min_samples, best_clusters, best_score


def plot_dbscan_results(pca_components, dbscan_clusters, eps, min_samples):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan_clusters, cmap='viridis', marker='o')
    plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()


def main():
    file_path = 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_data2.csv'
    df = load_data(file_path)
    features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                'Satisfaction Score']

    df = standardize_data(df, features)

    best_score = -1
    best_features = None
    best_clusters = None
    best_eps = None
    best_min_samples = None

    for subset in combinations(features, 5):  # Test combinations of 5 features at a time
        print(f"Testing feature subset: {subset}")
        pca_components = perform_pca(df, list(subset))  # Convert tuple to list
        eps, min_samples, clusters, score = dbscan_tuning(pca_components, subset)

        if score > best_score:
            best_score = score
            best_features = subset
            best_clusters = clusters
            best_eps = eps
            best_min_samples = min_samples

    if best_clusters is not None:
        print(f"\nBest Features: {best_features}")
        print(f"Best DBSCAN - eps: {best_eps:.3f}, min_samples: {best_min_samples}, Silhouette Score: {best_score:.3f}")
        plot_dbscan_results(pca_components, best_clusters, best_eps, best_min_samples)
    else:
        print("DBSCAN did not find valid clusters.")


if __name__ == "__main__":
    main()
