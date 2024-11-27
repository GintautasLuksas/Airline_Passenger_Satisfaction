import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class HistogramPaginator:
    def __init__(self, original_features, kmeans_labels, categorical_columns):
        """Initialize the paginator with data and clustering results."""
        self.original_features = original_features.copy()  # Avoid modifying the original DataFrame
        self.kmeans_labels = kmeans_labels
        self.numeric_columns = self.original_features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = categorical_columns  # Store categorical columns

    def plot_histograms(self):
        """Plot histograms for numerical features and bar plots for categorical features."""
        num_cols = len(self.numeric_columns)  # Total number of numeric columns
        rows, cols = self._get_grid_dimensions(num_cols)  # Calculate grid layout

        # Create subplots for the histograms with a smaller figure size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))  # Adjust size to make diagrams smaller
        axes = axes.flatten()  # Flatten for easy indexing

        # Plot histograms for numerical columns
        for idx, column in enumerate(self.numeric_columns):
            ax = axes[idx]
            for cluster in np.unique(self.kmeans_labels):
                cluster_data = self.original_features[self.original_features['Cluster'] == cluster]
                ax.hist(cluster_data[column], bins=10, alpha=0.6, label=f"Cluster {cluster}")  # Smaller number of bins
            ax.set_title(column, fontsize=8)  # Reduce title font size
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend(fontsize='x-small')

        # Hide unused axes
        for idx in range(len(self.numeric_columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

        # Plot bar charts for categorical columns like 'Gender'
        self.plot_categorical_barplots()

    def plot_categorical_barplots(self):
        """Plot bar charts for categorical columns."""
        for column in self.categorical_columns:
            plt.figure(figsize=(6, 4))
            for cluster in np.unique(self.kmeans_labels):
                cluster_data = self.original_features[self.original_features['Cluster'] == cluster]
                cluster_counts = cluster_data[column].value_counts()
                cluster_counts.plot(kind='bar', alpha=0.6, label=f"Cluster {cluster}")
            plt.title(f"Distribution of {column} by Cluster", fontsize=10)
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def _get_grid_dimensions(self, num_plots):
        """Calculate rows and columns for grid layout."""
        cols = min(5, num_plots)  # Up to 5 columns to fit more in one row
        rows = (num_plots + cols - 1) // cols  # Calculate required rows
        return rows, cols

def process_clusters_manual(n_clusters, pca_components, original_features):
    """Perform clustering with a specified number of clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(pca_components)
    original_features['Cluster'] = kmeans_clusters  # Add cluster labels
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']  # Include categorical features
    paginator = HistogramPaginator(original_features, kmeans_clusters, categorical_columns)
    paginator.plot_histograms()

def process_clusters_auto(pca_components, original_features):
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
    process_clusters_manual(best_n_clusters, pca_components, original_features)

def main():
    """Main function to execute clustering and visualization."""
    df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Cleaned_Data.csv')

    # Define features for clustering
    features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

    # Handle categorical features (Label Encoding)
    label_encoder = LabelEncoder()

    # Encoding categorical variables
    df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Encode Gender
    df['Customer Type'] = label_encoder.fit_transform(df['Customer Type'])  # Encode Customer Type
    df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])  # Encode Type of Travel
    df['Class'] = label_encoder.fit_transform(df['Class'])  # Encode Class

    # Data scaling and PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features].select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)

    # GUI creation
    root = tk.Tk()
    root.title("Clustering Options")

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Choose an option for clustering:", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)

    # Add a slider for the number of clusters
    ttk.Label(frame, text="Select number of clusters:").grid(row=1, column=0, sticky=tk.W, padx=10)
    cluster_slider = tk.Scale(frame, from_=2, to=15, orient=tk.HORIZONTAL)
    cluster_slider.set(3)  # Default number of clusters
    cluster_slider.grid(row=1, column=1, sticky=tk.W, padx=10)

    ttk.Button(frame, text="Run Manual Clustering", command=lambda: handle_manual_cluster(cluster_slider, pca_components, df[features])).grid(row=2, column=0, pady=10, sticky=tk.W, padx=10)
    ttk.Button(frame, text="Run Automatic Clustering", command=lambda: process_clusters_auto(pca_components, df[features])).grid(row=3, column=0, pady=10, sticky=tk.W, padx=10)
    ttk.Button(frame, text="Exit", command=root.quit).grid(row=4, column=0, pady=10, sticky=tk.W, padx=10)

    root.mainloop()

def handle_manual_cluster(cluster_slider, pca_components, df_features):
    """Handle manual clustering and validation."""
    n_clusters = cluster_slider.get()
    process_clusters_manual(n_clusters, pca_components, df_features)

if __name__ == "__main__":
    main()
