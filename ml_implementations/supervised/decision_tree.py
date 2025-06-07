"""
Decision Tree Implementation

This module implements a decision tree classifier and regressor from scratch using numpy.
The implementation includes both classification and regression capabilities, along with
various splitting criteria and pruning methods.

Author: ML Notes
Date: 2024
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from collections import Counter
import matplotlib.pyplot as plt

class Node:
    """
    A class representing a node in the decision tree.
    
    Attributes:
        feature_idx (int): Index of the feature used for splitting
        threshold (float): Threshold value for splitting
        left (Node): Left child node
        right (Node): Right child node
        value (float): Predicted value (for leaf nodes)
        impurity (float): Node impurity
        n_samples (int): Number of samples in the node
    """
    
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[float] = None,
        impurity: float = 0.0,
        n_samples: int = 0
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples

class DecisionTree:
    """
    A class to implement Decision Tree from scratch.
    
    This implementation includes:
    - Classification and Regression
    - Gini and Entropy criteria for classification
    - MSE criterion for regression
    - Pre-pruning (max_depth, min_samples_split)
    - Post-pruning (cost complexity pruning)
    
    Attributes:
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split
        min_samples_leaf (int): Minimum samples required in a leaf node
        criterion (str): Splitting criterion
        is_classification (bool): Whether the tree is for classification
        root (Node): Root node of the tree
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        is_classification: bool = True
    ):
        """
        Initialize the Decision Tree.
        
        Args:
            max_depth (int): Maximum depth of the tree
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required in a leaf node
            criterion (str): Splitting criterion ('gini', 'entropy', or 'mse')
            is_classification (bool): Whether the tree is for classification
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.is_classification = is_classification
        self.root = None
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate the impurity of a node.
        
        Args:
            y (np.ndarray): Target values
            
        Returns:
            float: Node impurity
        """
        if self.is_classification:
            if self.criterion == 'gini':
                return self._gini_impurity(y)
            else:  # entropy
                return self._entropy_impurity(y)
        else:  # regression
            return self._mse_impurity(y)
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity.
        
        Args:
            y (np.ndarray): Target values
            
        Returns:
            float: Gini impurity
        """
        counter = Counter(y)
        impurity = 1.0
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity
    
    def _entropy_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Entropy impurity.
        
        Args:
            y (np.ndarray): Target values
            
        Returns:
            float: Entropy impurity
        """
        counter = Counter(y)
        impurity = 0.0
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob * np.log2(prob)
        return impurity
    
    def _mse_impurity(self, y: np.ndarray) -> float:
        """
        Calculate MSE impurity.
        
        Args:
            y (np.ndarray): Target values
            
        Returns:
            float: MSE impurity
        """
        return np.mean((y - np.mean(y)) ** 2)
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split for a node.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            Tuple[Optional[int], Optional[float], float]: Best feature index, threshold, and impurity
        """
        n_samples, n_features = X.shape
        parent_impurity = self._calculate_impurity(y)
        best_impurity = float('inf')
        best_feature_idx = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                impurity = (np.sum(left_mask) * left_impurity +
                          np.sum(right_mask) * right_impurity) / n_samples
                
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_impurity
    
    def _create_leaf_node(self, y: np.ndarray) -> Node:
        """
        Create a leaf node.
        
        Args:
            y (np.ndarray): Target values
            
        Returns:
            Node: Leaf node
        """
        if self.is_classification:
            value = Counter(y).most_common(1)[0][0]
        else:
            value = np.mean(y)
        
        return Node(
            value=value,
            impurity=self._calculate_impurity(y),
            n_samples=len(y)
        )
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0
    ) -> Node:
        """
        Build the decision tree recursively.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            depth (int): Current depth of the tree
            
        Returns:
            Node: Root node of the subtree
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self._create_leaf_node(y)
        
        # Find best split
        best_feature_idx, best_threshold, best_impurity = self._find_best_split(X, y)
        
        # If no valid split found, create leaf node
        if best_feature_idx is None:
            return self._create_leaf_node(y)
        
        # Create internal node
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask
        
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
            impurity=best_impurity,
            n_samples=n_samples
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Train the decision tree.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            DecisionTree: Trained model
        """
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x: np.ndarray, node: Node) -> float:
        """
        Make a prediction for a single sample.
        
        Args:
            x (np.ndarray): Input features
            node (Node): Current node
            
        Returns:
            float: Predicted value
        """
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple samples.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the score of the model.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            float: Model score
        """
        predictions = self.predict(X)
        if self.is_classification:
            return np.mean(predictions == y)
        else:
            return 1 - np.mean((y - predictions) ** 2) / np.var(y)
    
    def _get_tree_depth(self, node: Node) -> int:
        """
        Get the depth of the tree.
        
        Args:
            node (Node): Current node
            
        Returns:
            int: Tree depth
        """
        if node is None:
            return 0
        return 1 + max(
            self._get_tree_depth(node.left),
            self._get_tree_depth(node.right)
        )
    
    def get_depth(self) -> int:
        """
        Get the depth of the tree.
        
        Returns:
            int: Tree depth
        """
        return self._get_tree_depth(self.root)
    
    def _get_n_leaves(self, node: Node) -> int:
        """
        Get the number of leaf nodes.
        
        Args:
            node (Node): Current node
            
        Returns:
            int: Number of leaf nodes
        """
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return self._get_n_leaves(node.left) + self._get_n_leaves(node.right)
    
    def get_n_leaves(self) -> int:
        """
        Get the number of leaf nodes.
        
        Returns:
            int: Number of leaf nodes
        """
        return self._get_n_leaves(self.root)

# Example usage
if __name__ == "__main__":
    # Generate sample data for classification
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create and train model
    model = DecisionTree(
        max_depth=5,
        min_samples_split=2,
        criterion='gini',
        is_classification=True
    )
    
    # Train the model
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate score
    accuracy = model.score(X, y)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tree Depth: {model.get_depth()}")
    print(f"Number of Leaves: {model.get_n_leaves()}") 