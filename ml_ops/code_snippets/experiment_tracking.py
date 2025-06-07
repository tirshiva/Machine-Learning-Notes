"""
Experiment Tracking with MLflow

This module demonstrates how to use MLflow for experiment tracking in ML projects.
It includes examples of tracking parameters, metrics, and artifacts.

Author: ML Notes
Date: 2024
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, Tuple

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split the iris dataset.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and test data
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, Any]
) -> RandomForestClassifier:
    """
    Train a Random Forest model with given parameters.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        params (Dict[str, Any]): Model parameters
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model (RandomForestClassifier): Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics

def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list
) -> None:
    """
    Plot feature importance.
    
    Args:
        model (RandomForestClassifier): Trained model
        feature_names (list): Names of features
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('feature_importance.png')
    plt.close()

def run_experiment(
    experiment_name: str,
    params: Dict[str, Any]
) -> None:
    """
    Run a complete experiment with MLflow tracking.
    
    Args:
        experiment_name (str): Name of the experiment
        params (Dict[str, Any]): Model parameters
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start run
    with mlflow.start_run(run_name=f"run_{params['n_estimators']}_trees"):
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = train_model(X_train, y_train, params)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Plot and log feature importance
        plot_feature_importance(model, load_iris().feature_names)
        mlflow.log_artifact('feature_importance.png')
        
        # Log additional information
        mlflow.log_text(
            "Model trained on iris dataset with Random Forest",
            "description.txt"
        )

def main():
    """
    Run multiple experiments with different parameters.
    """
    # Define parameter sets
    param_sets = [
        {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 3,
            'min_samples_leaf': 2
        },
        {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 4,
            'min_samples_leaf': 3
        }
    ]
    
    # Run experiments
    for params in param_sets:
        run_experiment("iris_classification", params)
    
    print("Experiments completed. Check MLflow UI for results.")

if __name__ == "__main__":
    main() 