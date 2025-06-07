"""
Model Monitoring and Drift Detection

This module implements model monitoring and drift detection functionality.
It includes performance monitoring, data drift detection, and concept drift detection.

Author: ML Notes
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    A class to monitor model performance and detect drift.
    
    This implementation includes:
    - Performance monitoring
    - Data drift detection
    - Concept drift detection
    - Statistical tests
    - Visualization
    - Alerting
    """
    
    def __init__(
        self,
        baseline_data: pd.DataFrame,
        baseline_predictions: np.ndarray,
        feature_names: List[str],
        target_name: str,
        drift_threshold: float = 0.05,
        window_size: int = 1000
    ):
        """
        Initialize the model monitor.
        
        Args:
            baseline_data (pd.DataFrame): Baseline data for drift detection
            baseline_predictions (np.ndarray): Baseline model predictions
            feature_names (List[str]): Names of features
            target_name (str): Name of target variable
            drift_threshold (float): Threshold for drift detection
            window_size (int): Size of sliding window for monitoring
        """
        self.baseline_data = baseline_data
        self.baseline_predictions = baseline_predictions
        self.feature_names = feature_names
        self.target_name = target_name
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        
        self.monitoring_history = []
        self.alerts = []
    
    def calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        feature: str
    ) -> Tuple[float, bool]:
        """
        Detect data drift for a single feature.
        
        Args:
            current_data (pd.DataFrame): Current data
            feature (str): Feature name
            
        Returns:
            Tuple[float, bool]: KS statistic and drift flag
        """
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(
            self.baseline_data[feature],
            current_data[feature]
        )
        
        # Check if drift is detected
        drift_detected = p_value < self.drift_threshold
        
        return ks_statistic, drift_detected
    
    def detect_concept_drift(
        self,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray,
        current_labels: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Detect concept drift using performance metrics.
        
        Args:
            current_data (pd.DataFrame): Current data
            current_predictions (np.ndarray): Current predictions
            current_labels (np.ndarray): Current true labels
            
        Returns:
            Tuple[float, bool]: Performance difference and drift flag
        """
        # Calculate performance metrics
        current_metrics = self.calculate_performance_metrics(
            current_labels,
            current_predictions
        )
        
        # Calculate baseline performance
        baseline_metrics = self.calculate_performance_metrics(
            self.baseline_data[self.target_name],
            self.baseline_predictions
        )
        
        # Calculate performance difference
        performance_diff = abs(
            current_metrics['accuracy'] - baseline_metrics['accuracy']
        )
        
        # Check if drift is detected
        drift_detected = performance_diff > self.drift_threshold
        
        return performance_diff, drift_detected
    
    def monitor_batch(
        self,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray,
        current_labels: np.ndarray
    ) -> Dict:
        """
        Monitor a batch of predictions.
        
        Args:
            current_data (pd.DataFrame): Current data
            current_predictions (np.ndarray): Current predictions
            current_labels (np.ndarray): Current true labels
            
        Returns:
            Dict: Monitoring results
        """
        # Initialize results
        results = {
            'timestamp': datetime.now(),
            'data_drift': {},
            'concept_drift': None,
            'performance_metrics': None,
            'alerts': []
        }
        
        # Check for data drift
        for feature in self.feature_names:
            ks_statistic, drift_detected = self.detect_data_drift(
                current_data,
                feature
            )
            
            results['data_drift'][feature] = {
                'ks_statistic': ks_statistic,
                'drift_detected': drift_detected
            }
            
            if drift_detected:
                alert = {
                    'type': 'data_drift',
                    'feature': feature,
                    'timestamp': datetime.now(),
                    'message': f'Data drift detected in feature {feature}'
                }
                results['alerts'].append(alert)
                self.alerts.append(alert)
        
        # Check for concept drift
        performance_diff, drift_detected = self.detect_concept_drift(
            current_data,
            current_predictions,
            current_labels
        )
        
        results['concept_drift'] = {
            'performance_diff': performance_diff,
            'drift_detected': drift_detected
        }
        
        if drift_detected:
            alert = {
                'type': 'concept_drift',
                'timestamp': datetime.now(),
                'message': 'Concept drift detected'
            }
            results['alerts'].append(alert)
            self.alerts.append(alert)
        
        # Calculate performance metrics
        results['performance_metrics'] = self.calculate_performance_metrics(
            current_labels,
            current_predictions
        )
        
        # Store results
        self.monitoring_history.append(results)
        
        return results
    
    def plot_drift_analysis(
        self,
        feature: str,
        window_size: int = 10
    ) -> None:
        """
        Plot drift analysis for a feature.
        
        Args:
            feature (str): Feature name
            window_size (int): Window size for smoothing
        """
        # Extract drift statistics
        timestamps = [r['timestamp'] for r in self.monitoring_history]
        ks_stats = [
            r['data_drift'][feature]['ks_statistic']
            for r in self.monitoring_history
        ]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, ks_stats, label='KS Statistic')
        plt.axhline(
            y=self.drift_threshold,
            color='r',
            linestyle='--',
            label='Drift Threshold'
        )
        plt.title(f'Drift Analysis for {feature}')
        plt.xlabel('Time')
        plt.ylabel('KS Statistic')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_performance_trends(self) -> None:
        """
        Plot performance metrics over time.
        """
        # Extract metrics
        timestamps = [r['timestamp'] for r in self.monitoring_history]
        metrics = {
            'accuracy': [r['performance_metrics']['accuracy'] for r in self.monitoring_history],
            'precision': [r['performance_metrics']['precision'] for r in self.monitoring_history],
            'recall': [r['performance_metrics']['recall'] for r in self.monitoring_history],
            'f1': [r['performance_metrics']['f1'] for r in self.monitoring_history]
        }
        
        # Create plot
        plt.figure(figsize=(12, 6))
        for metric, values in metrics.items():
            plt.plot(timestamps, values, label=metric)
        
        plt.title('Performance Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get alerts within a time range.
        
        Args:
            start_time (Optional[datetime]): Start time
            end_time (Optional[datetime]): End time
            
        Returns:
            List[Dict]: List of alerts
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.now()
        
        return [
            alert for alert in self.alerts
            if start_time <= alert['timestamp'] <= end_time
        ]
    
    def save_monitoring_results(self, filepath: str) -> None:
        """
        Save monitoring results to a file.
        
        Args:
            filepath (str): Path to save results
        """
        results = {
            'monitoring_history': [
                {
                    'timestamp': r['timestamp'].isoformat(),
                    'data_drift': r['data_drift'],
                    'concept_drift': r['concept_drift'],
                    'performance_metrics': r['performance_metrics'],
                    'alerts': r['alerts']
                }
                for r in self.monitoring_history
            ],
            'alerts': [
                {
                    'type': a['type'],
                    'timestamp': a['timestamp'].isoformat(),
                    'message': a['message']
                }
                for a in self.alerts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # Create baseline data
    baseline_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    baseline_data['target'] = (
        baseline_data['feature_0'] + baseline_data['feature_1'] > 0
    ).astype(int)
    
    # Create baseline predictions
    baseline_predictions = (
        baseline_data['feature_0'] + baseline_data['feature_1'] > 0
    ).astype(int)
    
    # Initialize monitor
    monitor = ModelMonitor(
        baseline_data=baseline_data,
        baseline_predictions=baseline_predictions,
        feature_names=[f'feature_{i}' for i in range(n_features)],
        target_name='target',
        drift_threshold=0.05
    )
    
    # Simulate monitoring
    for i in range(5):
        # Generate current data with drift
        current_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        if i > 2:  # Introduce drift
            current_data['feature_0'] += 2
        
        current_data['target'] = (
            current_data['feature_0'] + current_data['feature_1'] > 0
        ).astype(int)
        
        # Generate current predictions
        current_predictions = (
            current_data['feature_0'] + current_data['feature_1'] > 0
        ).astype(int)
        
        # Monitor batch
        results = monitor.monitor_batch(
            current_data,
            current_predictions,
            current_data['target']
        )
        
        print(f"\nBatch {i+1} Results:")
        print(f"Data Drift: {results['data_drift']}")
        print(f"Concept Drift: {results['concept_drift']}")
        print(f"Performance Metrics: {results['performance_metrics']}")
        print(f"Alerts: {results['alerts']}")
    
    # Plot results
    monitor.plot_drift_analysis('feature_0')
    monitor.plot_performance_trends()
    
    # Save results
    monitor.save_monitoring_results('monitoring_results.json') 