import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def process_clusters_manual(n_clusters, pca_components):
    """Perform clustering with a specified number of clusters."""
    plot_elbow_method(pca_components)
    plot_clusters(pca_components, n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(pca_components)
    score = silhouette_score(pca_components, kmeans_clusters)
    messagebox.showinfo("K-Means Result", f"K-Means Silhouette Score (n_clusters={n_clusters}): {score:.3f}")


def process_clusters_auto(pca_components):
    """Automatically find the best number of clusters."""
    best_score = -1
    best_n_clusters = 0

    for n_clusters in range(2, 16):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_clusters = kmeans.fit_predict(pca_components)
        score = silhouette_score(pca_components, kmeans_clusters)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    messagebox.showinfo(
        "Best Clustering",
        f"Best number of clusters: {best_n_clusters}\nSilhouette Score: {best_score:.3f}"
    )
    plot_clusters(pca_components, best_n_clusters)
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    kmeans.fit(pca_components)


def plot_elbow_method(pca_components):
    """Plot the elbow method graph."""
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
    """Plot clusters after applying K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(pca_components)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_clusters, cmap='viridis', marker='o')
    plt.title(f'K-Means Clustering (n_clusters={n_clusters})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()


def main():
    # Load dataset
    df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_Data2.csv')
    features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                'Satisfaction Score']

    if not all(col in df.columns for col in features):
        messagebox.showerror("Error", "Some required columns are missing from the dataset!")
        return

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df[features])

    # Create GUI
    def handle_manual_cluster():
        try:
            n_clusters = int(manual_input.get())
            process_clusters_manual(n_clusters, pca_components)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of clusters.")

    def handle_auto_cluster():
        process_clusters_auto(pca_components)

    root = tk.Tk()
    root.title("Clustering Options")

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Choose an option for clustering:", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)

    manual_input = tk.StringVar()
    ttk.Label(frame, text="Enter number of clusters (manual):").grid(row=1, column=0, sticky=tk.W, padx=10)
    ttk.Entry(frame, textvariable=manual_input).grid(row=1, column=1, sticky=tk.W)

    ttk.Button(frame, text="Run Manual Clustering", command=handle_manual_cluster).grid(row=2, column=0, pady=10, sticky=tk.W, padx=10)
    ttk.Button(frame, text="Run Automatic Clustering", command=handle_auto_cluster).grid(row=3, column=0, pady=10, sticky=tk.W, padx=10)
    ttk.Button(frame, text="Exit", command=root.quit).grid(row=4, column=0, pady=10, sticky=tk.W, padx=10)

    root.mainloop()


if __name__ == "__main__":
    main()
