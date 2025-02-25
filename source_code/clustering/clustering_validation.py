import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_clustering(csv_file, n_clusters=10):
    """
    Performs clustering using KMeans and visualizes the clusters in a 2D space after PCA.
    :param csv_file: Path to dataset CSV file
    :param n_clusters: Number of clusters (should match the number of digits: 10)
    """
    df = pd.read_csv(csv_file, header=None)
    
    # Separate features and labels
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Reduce dimensions using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot PCA clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Cluster Labels')
    plt.title("KMeans Clustering of Generated Digits (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
    
    # Check clustering quality by comparing with true labels
    cluster_purity = np.mean(cluster_labels == y)
    print(f"Clustering Purity Score: {cluster_purity:.4f}")

if __name__ == "__main__":
    perform_clustering("../data/dataset.csv")
