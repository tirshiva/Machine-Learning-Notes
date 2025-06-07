"""
Linear Regression Implementation

This module implements a simple linear regression model from scratch using numpy.
The implementation includes both batch gradient descent and stochastic gradient descent
optimization methods.

Author: ML Notes
Date: 2024
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class LinearRegression:
    """
    A class to implement Linear Regression from scratch.
    
    This implementation includes:
    - Batch Gradient Descent
    - Stochastic Gradient Descent
    - Regularization (L1/L2)
    - Learning rate scheduling
    - Early stopping
    
    Attributes:
        learning_rate (float): The learning rate for gradient descent
        n_iterations (int): Number of iterations for training
        regularization (str): Type of regularization ('l1', 'l2', or None)
        lambda_reg (float): Regularization strength
        weights (np.ndarray): Model weights
        bias (float): Model bias
        history (dict): Training history
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.1
    ):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Number of training iterations
            regularization (str): Type of regularization ('l1', 'l2', or None)
            lambda_reg (float): Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.history = {
            'loss': [],
            'weights': [],
            'bias': []
        }
    
    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters.
        
        Args:
            n_features (int): Number of features in the input data
        """
        self.weights = np.random.randn(n_features)
        self.bias = 0
    
    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: float
    ) -> float:
        """
        Compute the loss function (MSE) with optional regularization.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            weights (np.ndarray): Current weights
            bias (float): Current bias
            
        Returns:
            float: Computed loss value
        """
        predictions = np.dot(X, weights) + bias
        mse = np.mean((predictions - y) ** 2)
        
        # Add regularization if specified
        if self.regularization == 'l1':
            mse += self.lambda_reg * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            mse += self.lambda_reg * np.sum(weights ** 2)
            
        return mse
    
    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: float
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            weights (np.ndarray): Current weights
            bias (float): Current bias
            
        Returns:
            Tuple[np.ndarray, float]: Gradients for weights and bias
        """
        n_samples = X.shape[0]
        predictions = np.dot(X, weights) + bias
        
        # Compute gradients
        dw = (2/n_samples) * np.dot(X.T, (predictions - y))
        db = (2/n_samples) * np.sum(predictions - y)
        
        # Add regularization gradients
        if self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_reg * weights
            
        return dw, db
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'batch',
        batch_size: int = 32,
        early_stopping: bool = True,
        patience: int = 10
    ) -> 'LinearRegression':
        """
        Train the linear regression model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            method (str): Optimization method ('batch' or 'stochastic')
            batch_size (int): Batch size for stochastic gradient descent
            early_stopping (bool): Whether to use early stopping
            patience (int): Number of iterations to wait for improvement
            
        Returns:
            LinearRegression: Trained model
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(self.n_iterations):
            if method == 'batch':
                # Batch gradient descent
                dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            else:
                # Stochastic gradient descent
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                for j in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[j:j+batch_size]
                    y_batch = y_shuffled[j:j+batch_size]
                    dw, db = self._compute_gradients(X_batch, y_batch, self.weights, self.bias)
                    
                    # Update parameters
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            
            # Compute loss
            loss = self._compute_loss(X, y, self.weights, self.bias)
            
            # Store history
            self.history['loss'].append(loss)
            self.history['weights'].append(self.weights.copy())
            self.history['bias'].append(self.bias)
            
            # Early stopping
            if early_stopping:
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {i}")
                        break
            
            # Learning rate scheduling
            self.learning_rate *= 0.999
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the R² score of the model.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
    def plot_training_history(self) -> None:
        """
        Plot the training history (loss over iterations).
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'])
        plt.title('Training Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    
    # Create and train model
    model = LinearRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization='l2',
        lambda_reg=0.1
    )
    
    # Train the model
    model.fit(X, y, method='batch', early_stopping=True)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate score
    r2_score = model.score(X, y)
    print(f"R² Score: {r2_score:.4f}")
    
    # Plot training history
    model.plot_training_history() 