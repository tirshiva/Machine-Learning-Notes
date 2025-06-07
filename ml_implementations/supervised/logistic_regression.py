"""
Logistic Regression Implementation

This module implements a logistic regression model from scratch using numpy.
The implementation includes both binary and multi-class classification capabilities,
along with various optimization techniques and regularization methods.

Author: ML Notes
Date: 2024
"""

import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class LogisticRegression:
    """
    A class to implement Logistic Regression from scratch.
    
    This implementation includes:
    - Binary and Multi-class classification
    - Batch and Stochastic Gradient Descent
    - L1 and L2 Regularization
    - Learning rate scheduling
    - Early stopping
    - Cross-entropy loss
    
    Attributes:
        learning_rate (float): The learning rate for gradient descent
        n_iterations (int): Number of iterations for training
        regularization (str): Type of regularization ('l1', 'l2', or None)
        lambda_reg (float): Regularization strength
        weights (np.ndarray): Model weights
        bias (float): Model bias
        history (dict): Training history
        n_classes (int): Number of classes for multi-class classification
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.1,
        n_classes: int = 2
    ):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Number of training iterations
            regularization (str): Type of regularization ('l1', 'l2', or None)
            lambda_reg (float): Regularization strength
            n_classes (int): Number of classes for classification
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.n_classes = n_classes
        self.history = {
            'loss': [],
            'weights': [],
            'bias': []
        }
        self.encoder = OneHotEncoder(sparse=False) if n_classes > 2 else None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.
        
        Args:
            z (np.ndarray): Input values
            
        Returns:
            np.ndarray: Sigmoid of input values
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax function.
        
        Args:
            z (np.ndarray): Input values
            
        Returns:
            np.ndarray: Softmax probabilities
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters.
        
        Args:
            n_features (int): Number of features in the input data
        """
        if self.n_classes == 2:
            self.weights = np.random.randn(n_features)
            self.bias = 0
        else:
            self.weights = np.random.randn(n_features, self.n_classes)
            self.bias = np.zeros(self.n_classes)
    
    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray
    ) -> float:
        """
        Compute the cross-entropy loss with optional regularization.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            weights (np.ndarray): Current weights
            bias (np.ndarray): Current bias
            
        Returns:
            float: Computed loss value
        """
        n_samples = X.shape[0]
        
        if self.n_classes == 2:
            predictions = self._sigmoid(np.dot(X, weights) + bias)
            loss = -np.mean(y * np.log(predictions + 1e-15) + 
                          (1 - y) * np.log(1 - predictions + 1e-15))
        else:
            predictions = self._softmax(np.dot(X, weights) + bias)
            loss = -np.mean(np.sum(y * np.log(predictions + 1e-15), axis=1))
        
        # Add regularization
        if self.regularization == 'l1':
            loss += self.lambda_reg * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            loss += self.lambda_reg * np.sum(weights ** 2)
            
        return loss
    
    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for weights and bias.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            weights (np.ndarray): Current weights
            bias (np.ndarray): Current bias
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradients for weights and bias
        """
        n_samples = X.shape[0]
        
        if self.n_classes == 2:
            predictions = self._sigmoid(np.dot(X, weights) + bias)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
        else:
            predictions = self._softmax(np.dot(X, weights) + bias)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y, axis=0)
        
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
    ) -> 'LogisticRegression':
        """
        Train the logistic regression model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            method (str): Optimization method ('batch' or 'stochastic')
            batch_size (int): Batch size for stochastic gradient descent
            early_stopping (bool): Whether to use early stopping
            patience (int): Number of iterations to wait for improvement
            
        Returns:
            LogisticRegression: Trained model
        """
        n_samples, n_features = X.shape
        
        # Handle multi-class classification
        if self.n_classes > 2:
            y = self.encoder.fit_transform(y.reshape(-1, 1))
        
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if self.n_classes == 2:
            return self._sigmoid(np.dot(X, self.weights) + self.bias)
        else:
            return self._softmax(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        probabilities = self.predict_proba(X)
        if self.n_classes == 2:
            return (probabilities >= 0.5).astype(int)
        else:
            return np.argmax(probabilities, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy score of the model.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
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
    # Generate sample data for binary classification
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create and train model
    model = LogisticRegression(
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
    accuracy = model.score(X, y)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot training history
    model.plot_training_history() 