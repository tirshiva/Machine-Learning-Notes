"""
Model Selection Module

This module provides comprehensive model selection functionality including:
- Model comparison
- Ensemble methods
- Model stacking
- Model blending

Author: ML Notes
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Callable
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import logging
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelSelector:
    """
    A comprehensive model selection class that handles various selection tasks.
    """
    
    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        task_type: str = 'classification',
        cv: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the model selector.
        
        Args:
            models (Dict[str, BaseEstimator]): Dictionary of models to compare
            task_type (str): Type of task ('classification' or 'regression')
            cv (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
        """
        self.models = models
        self.task_type = task_type.lower()
        self.cv = cv
        self.random_state = random_state
        self.cv_splitter = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state
        )
        
        self.comparison_results = {}
        self.ensemble_model = None
    
    def compare_models(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            X: Features
            y: Target
            scoring: Scoring metric
            
        Returns:
            Dict[str, Dict[str, float]]: Comparison results
        """
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(
                model,
                X,
                y,
                cv=self.cv_splitter,
                scoring=scoring
            )
            
            results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
        
        self.comparison_results = results
        return results
    
    def plot_comparison(
        self,
        title: str = 'Model Comparison'
    ) -> None:
        """
        Plot model comparison results.
        
        Args:
            title (str): Plot title
        """
        if not self.comparison_results:
            raise ValueError('No comparison results available. Run compare_models first.')
        
        names = list(self.comparison_results.keys())
        means = [results['mean_score'] for results in self.comparison_results.values()]
        stds = [results['std_score'] for results in self.comparison_results.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, means, yerr=stds, capsize=5)
        plt.title(title)
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def create_ensemble(
        self,
        method: str = 'voting',
        weights: Optional[List[float]] = None
    ) -> None:
        """
        Create an ensemble of models.
        
        Args:
            method (str): Ensemble method ('voting' or 'averaging')
            weights (Optional[List[float]]): Weights for each model
        """
        if method == 'voting':
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            if self.task_type == 'classification':
                self.ensemble_model = VotingClassifier(
                    estimators=[(name, model) for name, model in self.models.items()],
                    voting='soft',
                    weights=weights
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=[(name, model) for name, model in self.models.items()],
                    weights=weights
                )
        else:  # averaging
            self.ensemble_model = AveragingEnsemble(
                models=list(self.models.values()),
                task_type=self.task_type,
                weights=weights
            )
    
    def create_stacking_ensemble(
        self,
        meta_model: BaseEstimator,
        use_probas: bool = True
    ) -> None:
        """
        Create a stacking ensemble.
        
        Args:
            meta_model (BaseEstimator): Meta-model for stacking
            use_probas (bool): Whether to use probabilities for classification
        """
        self.ensemble_model = StackingEnsemble(
            base_models=list(self.models.values()),
            meta_model=meta_model,
            task_type=self.task_type,
            cv=self.cv,
            use_probas=use_probas,
            random_state=self.random_state
        )
    
    def create_blending_ensemble(
        self,
        meta_model: BaseEstimator,
        validation_size: float = 0.2,
        use_probas: bool = True
    ) -> None:
        """
        Create a blending ensemble.
        
        Args:
            meta_model (BaseEstimator): Meta-model for blending
            validation_size (float): Size of validation set
            use_probas (bool): Whether to use probabilities for classification
        """
        self.ensemble_model = BlendingEnsemble(
            base_models=list(self.models.values()),
            meta_model=meta_model,
            task_type=self.task_type,
            validation_size=validation_size,
            use_probas=use_probas,
            random_state=self.random_state
        )
    
    def evaluate_ensemble(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.ensemble_model is None:
            raise ValueError('No ensemble model created. Run create_ensemble first.')
        
        # Fit and predict
        self.ensemble_model.fit(X, y)
        y_pred = self.ensemble_model.predict(X)
        
        # Calculate metrics
        metrics = {}
        if self.task_type == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            })
        else:
            metrics.update({
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            })
        
        return metrics

class AveragingEnsemble(BaseEstimator):
    """
    A simple averaging ensemble.
    """
    
    def __init__(
        self,
        models: List[BaseEstimator],
        task_type: str = 'classification',
        weights: Optional[List[float]] = None
    ):
        self.models = models
        self.task_type = task_type
        self.weights = weights
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'AveragingEnsemble':
        """
        Fit all models.
        """
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions by averaging.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        if self.weights is not None:
            return np.average(predictions, axis=0, weights=self.weights)
        return np.mean(predictions, axis=0)

class StackingEnsemble(BaseEstimator):
    """
    A stacking ensemble.
    """
    
    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: BaseEstimator,
        task_type: str = 'classification',
        cv: int = 5,
        use_probas: bool = True,
        random_state: int = 42
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.task_type = task_type
        self.cv = cv
        self.use_probas = use_probas
        self.random_state = random_state
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        """
        # Generate meta-features
        meta_features = self._generate_meta_features(X, y)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        
        # Fit base models on full dataset
        for model in self.base_models:
            model.fit(X, y)
        
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        """
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Make predictions using meta-model
        return self.meta_model.predict(meta_features)
    
    def _generate_meta_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> np.ndarray:
        """
        Generate meta-features using cross-validation.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        if self.task_type == 'classification' and self.use_probas:
            meta_features = np.zeros((n_samples, n_models * 2))
        else:
            meta_features = np.zeros((n_samples, n_models))
        
        cv = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            
            for i, model in enumerate(self.base_models):
                model.fit(X_train, y[train_idx] if y is not None else None)
                
                if self.task_type == 'classification' and self.use_probas:
                    probas = model.predict_proba(X_val)
                    meta_features[val_idx, i*2:(i+1)*2] = probas
                else:
                    meta_features[val_idx, i] = model.predict(X_val)
        
        return meta_features

class BlendingEnsemble(BaseEstimator):
    """
    A blending ensemble.
    """
    
    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: BaseEstimator,
        task_type: str = 'classification',
        validation_size: float = 0.2,
        use_probas: bool = True,
        random_state: int = 42
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.task_type = task_type
        self.validation_size = validation_size
        self.use_probas = use_probas
        self.random_state = random_state
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'BlendingEnsemble':
        """
        Fit the blending ensemble.
        """
        # Split data
        n_samples = X.shape[0]
        n_validation = int(n_samples * self.validation_size)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_validation:], indices[:n_validation]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X_val, y_val)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y_val)
        
        # Fit base models on full dataset
        for model in self.base_models:
            model.fit(X, y)
        
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions using the blending ensemble.
        """
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Make predictions using meta-model
        return self.meta_model.predict(meta_features)
    
    def _generate_meta_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> np.ndarray:
        """
        Generate meta-features using base models.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        if self.task_type == 'classification' and self.use_probas:
            meta_features = np.zeros((n_samples, n_models * 2))
        else:
            meta_features = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.base_models):
            if self.task_type == 'classification' and self.use_probas:
                probas = model.predict_proba(X)
                meta_features[:, i*2:(i+1)*2] = probas
            else:
                meta_features[:, i] = model.predict(X)
        
        return meta_features

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Initialize models
    models = {
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
        'svm': SVC(probability=True, random_state=42)
    }
    
    # Initialize model selector
    selector = ModelSelector(models, task_type='classification')
    
    # Compare models
    comparison_results = selector.compare_models(X, y)
    print("\nModel Comparison Results:")
    for name, results in comparison_results.items():
        print(f"{name}: {results['mean_score']:.3f} (+/- {results['std_score']:.3f})")
    
    # Plot comparison
    selector.plot_comparison()
    
    # Create and evaluate ensemble
    selector.create_ensemble(method='voting')
    ensemble_metrics = selector.evaluate_ensemble(X, y)
    print("\nEnsemble Metrics:", ensemble_metrics)
    
    # Create and evaluate stacking ensemble
    meta_model = LogisticRegression()
    selector.create_stacking_ensemble(meta_model)
    stacking_metrics = selector.evaluate_ensemble(X, y)
    print("\nStacking Ensemble Metrics:", stacking_metrics)
    
    # Create and evaluate blending ensemble
    selector.create_blending_ensemble(meta_model)
    blending_metrics = selector.evaluate_ensemble(X, y)
    print("\nBlending Ensemble Metrics:", blending_metrics) 