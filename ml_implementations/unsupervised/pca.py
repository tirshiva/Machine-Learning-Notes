"""
Principal Component Analysis (PCA) Implementation

This module implements Principal Component Analysis from scratch using numpy.
The implementation includes various methods for dimensionality reduction,
feature extraction, and visualization.

Author: ML Notes
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PCA:
    """
    A class to implement Principal Component Analysis from scratch.
    
    This implementation includes:
    - Data standardization
    - Covariance matrix computation
    - Eigenvalue decomposition
    - Dimensionality reduction
    - Feature extraction
    - Explained variance calculation
    - Visualization capabilities
    
    Attributes:
        n_components (int): Number of components to keep
        explained_variance_ (np.ndarray): Explained variance by each component
        explained_variance_ratio_ (np.ndarray): Ratio of explained variance
        components_ (np.ndarray): Principal components
        mean_ (np.ndarray): Mean of training data
        scaler (StandardScaler): Data scaler
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize PCA.
        
        Args:
            n_components (Optional[int]): Number of components to keep
            random_state (Optional[int]): Random seed
        """
        self.n_components = n_components
        self.random_state = random_state
        
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.mean_ = None
        self.scaler = StandardScaler()
    
    def _compute_covariance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Covariance matrix
        """
        return np.cov(X.T)
    
    def _compute_eigenvalues_eigenvectors(
        self,
        cov_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.
        
        Args:
            cov_matrix (np.ndarray): Covariance matrix
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Eigenvalues and eigenvectors
        """
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _determine_n_components(
        self,
        eigenvalues: np.ndarray,
        n_features: int
    ) -> int:
        """
        Determine number of components to keep.
        
        Args:
            eigenvalues (np.ndarray): Eigenvalues
            n_features (int): Number of features
            
        Returns:
            int: Number of components
        """
        if self.n_components is None:
            # Keep components that explain 95% of variance
            explained_variance_ratio = eigenvalues / eigenvalues.sum()
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            return np.argmax(cumulative_variance_ratio >= 0.95) + 1
        elif 0 < self.n_components < 1:
            # Keep components that explain specified variance ratio
            explained_variance_ratio = eigenvalues / eigenvalues.sum()
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            return np.argmax(cumulative_variance_ratio >= self.n_components) + 1
        else:
            return min(self.n_components, n_features)
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA to the data.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            PCA: Fitted model
        """
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        self.mean_ = self.scaler.mean_
        
        # Compute covariance matrix
        cov_matrix = self._compute_covariance_matrix(X_scaled)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self._compute_eigenvalues_eigenvectors(cov_matrix)
        
        # Determine number of components
        n_components = self._determine_n_components(eigenvalues, X.shape[1])
        
        # Store results
        self.components_ = eigenvectors[:, :n_components]
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigenvalues.sum()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal components.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Transformed data
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet")
        
        # Standardize data
        X_scaled = self.scaler.transform(X)
        
        # Project data onto principal components
        return np.dot(X_scaled, self.components_)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed (np.ndarray): Transformed data
            
        Returns:
            np.ndarray: Original data
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet")
        
        # Project back to original space
        X_original = np.dot(X_transformed, self.components_.T)
        
        # Inverse standardization
        return self.scaler.inverse_transform(X_original)
    
    def plot_explained_variance(
        self,
        title: str = "Explained Variance Ratio"
    ) -> None:
        """
        Plot explained variance ratio.
        
        Args:
            title (str): Plot title
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model has not been fitted yet")
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(self.explained_variance_ratio_) + 1),
            self.explained_variance_ratio_,
            'bo-'
        )
        plt.title(title)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        plt.show()
    
    def plot_cumulative_variance(
        self,
        title: str = "Cumulative Explained Variance"
    ) -> None:
        """
        Plot cumulative explained variance.
        
        Args:
            title (str): Plot title
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model has not been fitted yet")
        
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            'ro-'
        )
        plt.title(title)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
    
    def plot_components(
        self,
        X: np.ndarray,
        title: str = "PCA Components"
    ) -> None:
        """
        Plot first two principal components.
        
        Args:
            X (np.ndarray): Input data
            title (str): Plot title
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet")
        
        X_transformed = self.transform(X)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
        plt.title(title)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Create and fit model
    model = PCA(n_components=0.95)  # Keep 95% of variance
    X_transformed = model.fit_transform(X)
    
    # Print results
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {model.explained_variance_ratio_}")
    
    # Plot results
    model.plot_explained_variance()
    model.plot_cumulative_variance()
    model.plot_components(X) 