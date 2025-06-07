"""
K-Means Clustering Implementation

This module implements the K-Means clustering algorithm from scratch using numpy.
The implementation includes various initialization methods, distance metrics, and
visualization capabilities.

Author: ML Notes
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class KMeans:
    """
    A class to implement K-Means clustering from scratch.
    
    This implementation includes:
    - Multiple initialization methods (random, k-means++)
    - Various distance metrics (Euclidean, Manhattan, Cosine)
    - Convergence criteria
    - Cluster visualization
    - Performance metrics (silhouette score, inertia)
    
    Attributes:
        n_clusters (int): Number of clusters
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        init (str): Initialization method
        metric (str): Distance metric
        random_state (int): Random seed
        centroids (np.ndarray): Cluster centroids
        labels (np.ndarray): Cluster assignments
        inertia (float): Sum of squared distances
        history (List[float]): History of inertia values
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        init: str = 'k-means++',
        metric: str = 'euclidean',
        random_state: Optional[int] = None
    ):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters (int): Number of clusters
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
            init (str): Initialization method ('random' or 'k-means++')
            metric (str): Distance metric ('euclidean', 'manhattan', or 'cosine')
            random_state (Optional[int]): Random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.metric = metric
        self.random_state = random_state
        
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.history = []
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centroids.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Initial centroids
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.init == 'random':
            # Random initialization
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]
        
        elif self.init == 'k-means++':
            # K-means++ initialization
            centroids = [X[np.random.randint(X.shape[0])]]
            
            for _ in range(1, self.n_clusters):
                # Calculate distances to nearest centroid
                distances = np.array([
                    min(self._compute_distance(x, c) for c in centroids)
                    for x in X
                ])
                
                # Convert distances to probabilities
                probs = distances / distances.sum()
                
                # Choose next centroid
                next_centroid = X[np.random.choice(X.shape[0], p=probs)]
                centroids.append(next_centroid)
            
            return np.array(centroids)
        
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
    
    def _compute_distance(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Compute distance between two points.
        
        Args:
            x (np.ndarray): First point
            y (np.ndarray): Second point
            
        Returns:
            float: Distance between points
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        elif self.metric == 'cosine':
            return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_distances(
        self,
        X: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between points and centroids.
        
        Args:
            X (np.ndarray): Input data
            centroids (np.ndarray): Cluster centroids
            
        Returns:
            np.ndarray: Distance matrix
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for i in range(n_samples):
            for j in range(n_clusters):
                distances[i, j] = self._compute_distance(X[i], centroids[j])
        
        return distances
    
    def _assign_clusters(
        self,
        X: np.ndarray,
        centroids: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Assign points to nearest clusters.
        
        Args:
            X (np.ndarray): Input data
            centroids (np.ndarray): Cluster centroids
            
        Returns:
            Tuple[np.ndarray, float]: Cluster assignments and inertia
        """
        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1))
        
        return labels, inertia
    
    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Update cluster centroids.
        
        Args:
            X (np.ndarray): Input data
            labels (np.ndarray): Cluster assignments
            
        Returns:
            np.ndarray: Updated centroids
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)
        
        return centroids
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-Means clustering to the data.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            KMeans: Fitted model
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Initialize variables
        prev_inertia = float('inf')
        self.history = []
        
        # Main loop
        for iteration in range(self.max_iter):
            # Assign points to clusters
            self.labels, self.inertia = self._assign_clusters(X, self.centroids)
            
            # Store history
            self.history.append(self.inertia)
            
            # Check convergence
            if abs(prev_inertia - self.inertia) < self.tol:
                break
            
            # Update centroids
            self.centroids = self._update_centroids(X, self.labels)
            prev_inertia = self.inertia
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Cluster assignments
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet")
        
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def score(self, X: np.ndarray) -> float:
        """
        Calculate the silhouette score.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            float: Silhouette score
        """
        if self.labels is None:
            raise ValueError("Model has not been fitted yet")
        
        return silhouette_score(X, self.labels)
    
    def plot_clusters(
        self,
        X: np.ndarray,
        title: str = "K-Means Clustering"
    ) -> None:
        """
        Plot clusters (for 2D data only).
        
        Args:
            X (np.ndarray): Input data
            title (str): Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(
            self.centroids[:, 0],
            self.centroids[:, 1],
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Centroids'
        )
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
    
    def plot_inertia_history(self, title: str = "Inertia History") -> None:
        """
        Plot the history of inertia values.
        
        Args:
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 2)
    
    # Create and train model
    model = KMeans(
        n_clusters=3,
        init='k-means++',
        metric='euclidean',
        random_state=42
    )
    
    # Fit the model
    model.fit(X)
    
    # Get cluster assignments
    labels = model.predict(X)
    
    # Calculate silhouette score
    score = model.score(X)
    print(f"Silhouette Score: {score:.4f}")
    
    # Plot clusters
    model.plot_clusters(X)
    
    # Plot inertia history
    model.plot_inertia_history() 