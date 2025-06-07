"""
Support Vector Machine (SVM) Implementation

This module implements a Support Vector Machine classifier from scratch using numpy.
The implementation includes both linear and kernel-based SVMs, along with various
optimization techniques and kernel functions.

Author: ML Notes
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple, Callable
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

class SVM:
    """
    A class to implement Support Vector Machine from scratch.
    
    This implementation includes:
    - Linear and Kernel-based SVM
    - Various kernel functions (RBF, Polynomial, Linear)
    - Soft margin classification
    - SMO (Sequential Minimal Optimization) algorithm
    - Support vector selection
    - Decision function calculation
    
    Attributes:
        C (float): Regularization parameter
        kernel (str): Kernel function type
        gamma (float): RBF kernel parameter
        degree (int): Polynomial kernel degree
        coef0 (float): Polynomial kernel coefficient
        alpha (np.ndarray): Lagrange multipliers
        support_vectors (np.ndarray): Support vectors
        support_vector_labels (np.ndarray): Labels of support vectors
        b (float): Bias term
        kernel_func (Callable): Kernel function
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: float = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000
    ):
        """
        Initialize the SVM.
        
        Args:
            C (float): Regularization parameter
            kernel (str): Kernel function type ('linear', 'rbf', or 'poly')
            gamma (float): RBF kernel parameter
            degree (int): Polynomial kernel degree
            coef0 (float): Polynomial kernel coefficient
            tol (float): Tolerance for stopping criterion
            max_iter (int): Maximum number of iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
        self.kernel_func = None
    
    def _initialize_kernel(self, X: np.ndarray) -> None:
        """
        Initialize the kernel function.
        
        Args:
            X (np.ndarray): Training features
        """
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X.shape[1] * X.var())
        
        if self.kernel == 'linear':
            self.kernel_func = lambda x1, x2: linear_kernel(x1, x2)
        elif self.kernel == 'rbf':
            self.kernel_func = lambda x1, x2: rbf_kernel(x1, x2, gamma=self.gamma)
        elif self.kernel == 'poly':
            self.kernel_func = lambda x1, x2: polynomial_kernel(
                x1, x2, degree=self.degree, gamma=self.gamma, coef0=self.coef0
            )
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Kernel matrix
        """
        return self.kernel_func(X, X)
    
    def _compute_eta(self, i: int, j: int, K: np.ndarray) -> float:
        """
        Compute eta for SMO algorithm.
        
        Args:
            i (int): First index
            j (int): Second index
            K (np.ndarray): Kernel matrix
            
        Returns:
            float: Eta value
        """
        return 2 * K[i, j] - K[i, i] - K[j, j]
    
    def _compute_L_H(
        self,
        i: int,
        j: int,
        y: np.ndarray,
        alpha: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute L and H for SMO algorithm.
        
        Args:
            i (int): First index
            j (int): Second index
            y (np.ndarray): Target values
            alpha (np.ndarray): Lagrange multipliers
            
        Returns:
            Tuple[float, float]: L and H values
        """
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(self.C, self.C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - self.C)
            H = min(self.C, alpha[i] + alpha[j])
        return L, H
    
    def _compute_E(
        self,
        i: int,
        X: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        b: float,
        K: np.ndarray
    ) -> float:
        """
        Compute error for SMO algorithm.
        
        Args:
            i (int): Sample index
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            alpha (np.ndarray): Lagrange multipliers
            b (float): Bias term
            K (np.ndarray): Kernel matrix
            
        Returns:
            float: Error value
        """
        return self._decision_function(X[i], X, y, alpha, b, K) - y[i]
    
    def _decision_function(
        self,
        x: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        b: float,
        K: np.ndarray
    ) -> float:
        """
        Compute decision function.
        
        Args:
            x (np.ndarray): Input features
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            alpha (np.ndarray): Lagrange multipliers
            b (float): Bias term
            K (np.ndarray): Kernel matrix
            
        Returns:
            float: Decision function value
        """
        if self.kernel == 'linear':
            return np.dot(x, np.sum(alpha * y * X.T, axis=1)) + b
        else:
            return np.sum(alpha * y * self.kernel_func(X, x.reshape(1, -1))) + b
    
    def _update_b(
        self,
        i: int,
        j: int,
        X: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        b: float,
        K: np.ndarray
    ) -> float:
        """
        Update bias term.
        
        Args:
            i (int): First index
            j (int): Second index
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            alpha (np.ndarray): Lagrange multipliers
            b (float): Current bias term
            K (np.ndarray): Kernel matrix
            
        Returns:
            float: Updated bias term
        """
        b1 = b - self._compute_E(i, X, y, alpha, b, K) - y[i] * (alpha[i] - self.alpha[i]) * K[i, i] - y[j] * (alpha[j] - self.alpha[j]) * K[i, j]
        b2 = b - self._compute_E(j, X, y, alpha, b, K) - y[i] * (alpha[i] - self.alpha[i]) * K[i, j] - y[j] * (alpha[j] - self.alpha[j]) * K[j, j]
        return (b1 + b2) / 2
    
    def _smo_step(
        self,
        i: int,
        j: int,
        X: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        b: float,
        K: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Perform one step of SMO algorithm.
        
        Args:
            i (int): First index
            j (int): Second index
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            alpha (np.ndarray): Lagrange multipliers
            b (float): Bias term
            K (np.ndarray): Kernel matrix
            
        Returns:
            Tuple[np.ndarray, float, bool]: Updated alpha, bias, and whether step was successful
        """
        if i == j:
            return alpha, b, False
        
        Ei = self._compute_E(i, X, y, alpha, b, K)
        Ej = self._compute_E(j, X, y, alpha, b, K)
        
        L, H = self._compute_L_H(i, j, y, alpha)
        if L == H:
            return alpha, b, False
        
        eta = self._compute_eta(i, j, K)
        if eta >= 0:
            return alpha, b, False
        
        alpha_j = alpha[j] - y[j] * (Ei - Ej) / eta
        alpha_j = np.clip(alpha_j, L, H)
        
        if abs(alpha_j - alpha[j]) < self.tol:
            return alpha, b, False
        
        alpha_i = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j)
        
        new_alpha = alpha.copy()
        new_alpha[i] = alpha_i
        new_alpha[j] = alpha_j
        
        new_b = self._update_b(i, j, X, y, new_alpha, b, K)
        
        return new_alpha, new_b, True
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Train the SVM.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            SVM: Trained model
        """
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        
        self._initialize_kernel(X)
        K = self._compute_kernel_matrix(X)
        
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        n_changed = 0
        examine_all = True
        iteration = 0
        
        while (n_changed > 0 or examine_all) and iteration < self.max_iter:
            n_changed = 0
            
            if examine_all:
                for i in range(n_samples):
                    n_changed += self._examine_example(i, X, y, K)
            else:
                for i in range(n_samples):
                    if 0 < self.alpha[i] < self.C:
                        n_changed += self._examine_example(i, X, y, K)
            
            if examine_all:
                examine_all = False
            elif n_changed == 0:
                examine_all = True
            
            iteration += 1
        
        # Store support vectors
        sv_mask = self.alpha > self.tol
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.alpha = self.alpha[sv_mask]
        
        return self
    
    def _examine_example(
        self,
        i: int,
        X: np.ndarray,
        y: np.ndarray,
        K: np.ndarray
    ) -> int:
        """
        Examine and update an example in SMO algorithm.
        
        Args:
            i (int): Example index
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            K (np.ndarray): Kernel matrix
            
        Returns:
            int: Number of changes made
        """
        Ei = self._compute_E(i, X, y, self.alpha, self.b, K)
        
        if ((y[i] * Ei < -self.tol and self.alpha[i] < self.C) or
            (y[i] * Ei > self.tol and self.alpha[i] > 0)):
            
            # Find second alpha to optimize
            j = self._select_second_alpha(i, X, y, K)
            if j is not None:
                new_alpha, new_b, success = self._smo_step(
                    i, j, X, y, self.alpha, self.b, K
                )
                if success:
                    self.alpha = new_alpha
                    self.b = new_b
                    return 1
        
        return 0
    
    def _select_second_alpha(
        self,
        i: int,
        X: np.ndarray,
        y: np.ndarray,
        K: np.ndarray
    ) -> Optional[int]:
        """
        Select second alpha for optimization.
        
        Args:
            i (int): First alpha index
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            K (np.ndarray): Kernel matrix
            
        Returns:
            Optional[int]: Second alpha index
        """
        Ei = self._compute_E(i, X, y, self.alpha, self.b, K)
        n_samples = X.shape[0]
        
        # Try to find alpha that maximizes |E1 - E2|
        max_delta_E = 0
        j = None
        
        for k in range(n_samples):
            if k != i:
                Ek = self._compute_E(k, X, y, self.alpha, self.b, K)
                delta_E = abs(Ei - Ek)
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    j = k
        
        return j
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.support_vectors is None:
            raise ValueError("Model has not been fitted yet")
        
        decisions = np.array([
            self._decision_function(
                x, self.support_vectors, self.support_vector_labels,
                self.alpha, self.b, self._compute_kernel_matrix(self.support_vectors)
            )
            for x in X
        ])
        
        return np.where(decisions >= 0, 1, 0)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Decision function values
        """
        if self.support_vectors is None:
            raise ValueError("Model has not been fitted yet")
        
        return np.array([
            self._decision_function(
                x, self.support_vectors, self.support_vector_labels,
                self.alpha, self.b, self._compute_kernel_matrix(self.support_vectors)
            )
            for x in X
        ])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy score.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_decision_boundary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        title: str = "SVM Decision Boundary"
    ) -> None:
        """
        Plot decision boundary (for 2D data only).
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            title (str): Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data")
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02)
        )
        
        # Predict for mesh grid points
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        if self.support_vectors is not None:
            plt.scatter(
                self.support_vectors[:, 0],
                self.support_vectors[:, 1],
                c='red',
                marker='x',
                s=100,
                linewidths=1,
                label='Support Vectors'
            )
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create and train model
    model = SVM(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        tol=1e-3,
        max_iter=1000
    )
    
    # Train the model
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate score
    accuracy = model.score(X, y)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot decision boundary
    model.plot_decision_boundary(X, y) 