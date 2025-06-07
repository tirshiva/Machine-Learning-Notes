"""
Random Forest Implementation

This module implements a Random Forest classifier and regressor from scratch using numpy.
The implementation includes both classification and regression capabilities, along with
various ensemble methods and feature importance calculation.

Author: ML Notes
Date: 2024
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from collections import Counter
import matplotlib.pyplot as plt
from .decision_tree import DecisionTree

class RandomForest:
    """
    A class to implement Random Forest from scratch.
    
    This implementation includes:
    - Classification and Regression
    - Bootstrap sampling
    - Feature subsampling
    - Majority voting for classification
    - Mean prediction for regression
    - Feature importance calculation
    - Out-of-bag error estimation
    
    Attributes:
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of each tree
        min_samples_split (int): Minimum samples required to split
        min_samples_leaf (int): Minimum samples required in a leaf node
        max_features (Union[int, float, str]): Number of features to consider for splitting
        bootstrap (bool): Whether to use bootstrap sampling
        criterion (str): Splitting criterion
        is_classification (bool): Whether the forest is for classification
        trees (List[DecisionTree]): List of decision trees
        feature_importances_ (np.ndarray): Feature importance scores
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[int, float, str] = 'sqrt',
        bootstrap: bool = True,
        criterion: str = 'gini',
        is_classification: bool = True
    ):
        """
        Initialize the Random Forest.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of each tree
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required in a leaf node
            max_features (Union[int, float, str]): Number of features to consider for splitting
            bootstrap (bool): Whether to use bootstrap sampling
            criterion (str): Splitting criterion
            is_classification (bool): Whether the forest is for classification
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.is_classification = is_classification
        self.trees = []
        self.feature_importances_ = None
    
    def _get_max_features(self, n_features: int) -> int:
        """
        Get the number of features to consider for splitting.
        
        Args:
            n_features (int): Total number of features
            
        Returns:
            int: Number of features to consider
        """
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        else:
            return n_features
    
    def _bootstrap_sample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a bootstrap sample of the data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Bootstrap sample and out-of-bag indices
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        oob_indices = np.array([i for i in range(n_samples) if i not in indices])
        return X[indices], y[indices], oob_indices
    
    def _calculate_feature_importance(self) -> None:
        """
        Calculate feature importance scores.
        """
        n_features = self.trees[0].root.n_samples
        importances = np.zeros(n_features)
        
        for tree in self.trees:
            importances += self._calculate_tree_importance(tree)
        
        self.feature_importances_ = importances / len(self.trees)
    
    def _calculate_tree_importance(self, tree: DecisionTree) -> np.ndarray:
        """
        Calculate feature importance for a single tree.
        
        Args:
            tree (DecisionTree): Decision tree
            
        Returns:
            np.ndarray: Feature importance scores
        """
        def _calculate_node_importance(node: 'Node') -> Tuple[np.ndarray, float]:
            if node is None:
                return np.zeros(n_features), 0.0
            
            if node.left is None and node.right is None:
                return np.zeros(n_features), 0.0
            
            left_imp, left_impurity = _calculate_node_importance(node.left)
            right_imp, right_impurity = _calculate_node_importance(node.right)
            
            importance = np.zeros(n_features)
            importance[node.feature_idx] = (
                node.impurity -
                (node.left.n_samples * left_impurity +
                 node.right.n_samples * right_impurity) / node.n_samples
            )
            
            return importance + left_imp + right_imp, node.impurity
        
        n_features = tree.root.n_samples
        importance, _ = _calculate_node_importance(tree.root)
        return importance
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Train the random forest.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            RandomForest: Trained model
        """
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        
        self.trees = []
        oob_predictions = np.zeros((n_samples, self.n_estimators))
        oob_counts = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                is_classification=self.is_classification
            )
            
            if self.bootstrap:
                # Bootstrap sampling
                X_boot, y_boot, oob_indices = self._bootstrap_sample(X, y)
                tree.fit(X_boot, y_boot)
                
                # Out-of-bag predictions
                if len(oob_indices) > 0:
                    oob_pred = tree.predict(X[oob_indices])
                    oob_predictions[oob_indices, i] = oob_pred
                    oob_counts[oob_indices] += 1
            else:
                tree.fit(X, y)
            
            self.trees.append(tree)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Calculate out-of-bag error
        if self.bootstrap:
            oob_mask = oob_counts > 0
            oob_pred = np.zeros_like(y, dtype=float)
            
            if self.is_classification:
                for i in range(n_samples):
                    if oob_mask[i]:
                        preds = oob_predictions[i, oob_counts[i] > 0]
                        oob_pred[i] = Counter(preds).most_common(1)[0][0]
            else:
                for i in range(n_samples):
                    if oob_mask[i]:
                        preds = oob_predictions[i, oob_counts[i] > 0]
                        oob_pred[i] = np.mean(preds)
            
            self.oob_score_ = self.score(X[oob_mask], y[oob_mask])
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the random forest.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        if self.is_classification:
            # Majority voting for classification
            return np.array([
                Counter(pred).most_common(1)[0][0]
                for pred in predictions.T
            ])
        else:
            # Mean prediction for regression
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classification only).
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification")
        
        predictions = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        n_classes = len(np.unique(predictions))
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j in range(n_classes):
                probabilities[i, j] = np.mean(predictions[:, i] == j)
        
        return probabilities
    
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
    
    def plot_feature_importance(self) -> None:
        """
        Plot feature importance scores.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been fitted yet")
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.feature_importances_)), self.feature_importances_)
        plt.title('Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data for classification
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create and train model
    model = RandomForest(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        max_features='sqrt',
        bootstrap=True,
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
    if hasattr(model, 'oob_score_'):
        print(f"Out-of-bag Score: {model.oob_score_:.4f}")
    
    # Plot feature importance
    model.plot_feature_importance() 