"""
Model Evaluation Module

This module provides comprehensive model evaluation functionality including:
- Cross-validation techniques
- Hyperparameter tuning
- Model selection criteria
- Performance metrics
- Learning curves
- Validation curves

Author: ML Notes
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Callable
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold,
    learning_curve, validation_curve,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
import logging
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A comprehensive model evaluation class that handles various evaluation tasks.
    """
    
    def __init__(
        self,
        model: object,
        task_type: str = 'classification',
        cv: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model (object): The model to evaluate
            task_type (str): Type of task ('classification' or 'regression')
            cv (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
        """
        self.model = model
        self.task_type = task_type.lower()
        self.cv = cv
        self.random_state = random_state
        
        # Initialize cross-validation
        if self.task_type == 'classification':
            self.cv_splitter = StratifiedKFold(
                n_splits=cv,
                shuffle=True,
                random_state=random_state
            )
        else:
            self.cv_splitter = KFold(
                n_splits=cv,
                shuffle=True,
                random_state=random_state
            )
    
    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            scoring: Scoring metric
            
        Returns:
            Dict[str, float]: Cross-validation results
        """
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=self.cv_splitter,
            scoring=scoring
        )
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def evaluate_performance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Training features
            y: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Fit model
        self.model.fit(X, y)
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = (
            self.model.predict_proba(X)[:, 1]
            if hasattr(self.model, 'predict_proba')
            else None
        )
        
        # Calculate metrics
        metrics = {}
        
        if self.task_type == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            })
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            })
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            y_test_pred = self.model.predict(X_test)
            y_test_pred_proba = (
                self.model.predict_proba(X_test)[:, 1]
                if hasattr(self.model, 'predict_proba')
                else None
            )
            
            test_metrics = {}
            
            if self.task_type == 'classification':
                test_metrics.update({
                    'test_accuracy': accuracy_score(y_test, y_test_pred),
                    'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
                    'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
                    'test_f1': f1_score(y_test, y_test_pred, average='weighted')
                })
                
                if y_test_pred_proba is not None:
                    test_metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_pred_proba)
            
            else:  # regression
                test_metrics.update({
                    'test_mse': mean_squared_error(y_test, y_test_pred),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    'test_mae': mean_absolute_error(y_test, y_test_pred),
                    'test_r2': r2_score(y_test, y_test_pred)
                })
            
            metrics.update(test_metrics)
        
        return metrics
    
    def plot_learning_curve(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        title: str = 'Learning Curve'
    ) -> None:
        """
        Plot learning curve.
        
        Args:
            X: Features
            y: Target
            title: Plot title
        """
        train_sizes, train_scores, val_scores = learning_curve(
            self.model,
            X,
            y,
            cv=self.cv_splitter,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1
        )
        plt.plot(train_sizes, val_mean, label='Cross-validation score')
        plt.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1
        )
        plt.title(title)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def plot_validation_curve(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        param_name: str,
        param_range: np.ndarray,
        title: str = 'Validation Curve'
    ) -> None:
        """
        Plot validation curve.
        
        Args:
            X: Features
            y: Target
            param_name: Parameter name
            param_range: Parameter range
            title: Plot title
        """
        train_scores, val_scores = validation_curve(
            self.model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=self.cv_splitter,
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, label='Training score')
        plt.fill_between(
            param_range,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1
        )
        plt.plot(param_range, val_mean, label='Cross-validation score')
        plt.fill_between(
            param_range,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1
        )
        plt.title(title)
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        title: str = 'Confusion Matrix'
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
        """
        if self.task_type != 'classification':
            raise ValueError('Confusion matrix is only for classification tasks')
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        title: str = 'Feature Importance'
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            title: Plot title
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError('Model does not have feature_importances_ attribute')
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def hyperparameter_tuning(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        param_grid: Dict,
        method: str = 'grid',
        n_iter: int = 100
    ) -> Dict:
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            param_grid: Parameter grid
            method: Tuning method ('grid' or 'random')
            n_iter: Number of iterations for random search
            
        Returns:
            Dict: Best parameters and score
        """
        if method == 'grid':
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=self.cv_splitter,
                n_jobs=-1,
                scoring='accuracy' if self.task_type == 'classification' else 'r2'
            )
        else:  # random
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=self.cv_splitter,
                n_jobs=-1,
                scoring='accuracy' if self.task_type == 'classification' else 'r2'
            )
        
        search.fit(X, y)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_
        }

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Initialize model and evaluator
    model = RandomForestClassifier(random_state=42)
    evaluator = ModelEvaluator(model, task_type='classification')
    
    # Cross-validation
    cv_results = evaluator.cross_validate(X, y)
    print("\nCross-validation Results:")
    print(f"Mean Score: {cv_results['mean_score']:.3f} (+/- {cv_results['std_score']:.3f})")
    
    # Performance evaluation
    metrics = evaluator.evaluate_performance(X, y)
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Plot learning curve
    evaluator.plot_learning_curve(X, y)
    
    # Plot validation curve
    param_range = np.arange(1, 11)
    evaluator.plot_validation_curve(
        X,
        y,
        param_name='max_depth',
        param_range=param_range
    )
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    tuning_results = evaluator.hyperparameter_tuning(
        X,
        y,
        param_grid,
        method='grid'
    )
    
    print("\nBest Parameters:", tuning_results['best_params'])
    print("Best Score:", tuning_results['best_score']) 