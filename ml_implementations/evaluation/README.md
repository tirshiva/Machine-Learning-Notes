# Model Evaluation Module

This module provides comprehensive model evaluation functionality for machine learning projects. It includes various techniques for evaluating model performance, tuning hyperparameters, and analyzing model behavior.

## Features

### 1. Cross-Validation
- K-Fold cross-validation
- Stratified K-Fold cross-validation
- Custom scoring metrics
- Cross-validation results analysis

### 2. Performance Metrics
- Classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
  - Confusion Matrix
- Regression metrics:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)

### 3. Learning Curves
- Training vs. validation score
- Bias-variance analysis
- Overfitting detection
- Underfitting detection

### 4. Validation Curves
- Parameter tuning visualization
- Optimal parameter selection
- Model complexity analysis

### 5. Hyperparameter Tuning
- Grid Search
- Random Search
- Custom parameter grids
- Best parameter selection

### 6. Feature Importance
- Feature ranking
- Importance visualization
- Feature selection guidance

## Usage

### Basic Usage

```python
from ml_implementations.evaluation.model_evaluation import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier

# Initialize model and evaluator
model = RandomForestClassifier()
evaluator = ModelEvaluator(model, task_type='classification')

# Cross-validation
cv_results = evaluator.cross_validate(X, y)
print(f"Mean Score: {cv_results['mean_score']:.3f}")

# Performance evaluation
metrics = evaluator.evaluate_performance(X, y)
print("Performance Metrics:", metrics)
```

### Learning and Validation Curves

```python
# Plot learning curve
evaluator.plot_learning_curve(X, y, title='Model Learning Curve')

# Plot validation curve
param_range = np.arange(1, 11)
evaluator.plot_validation_curve(
    X,
    y,
    param_name='max_depth',
    param_range=param_range,
    title='Validation Curve for max_depth'
)
```

### Hyperparameter Tuning

```python
# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
tuning_results = evaluator.hyperparameter_tuning(
    X,
    y,
    param_grid,
    method='grid'
)

print("Best Parameters:", tuning_results['best_params'])
print("Best Score:", tuning_results['best_score'])
```

## Best Practices

1. **Cross-Validation**
   - Use appropriate number of folds
   - Consider stratified sampling for classification
   - Use multiple scoring metrics

2. **Performance Evaluation**
   - Use multiple metrics
   - Consider domain-specific metrics
   - Evaluate on both training and test sets

3. **Hyperparameter Tuning**
   - Start with broad parameter ranges
   - Use appropriate search method
   - Consider computational cost

4. **Learning Curves**
   - Monitor training and validation scores
   - Identify overfitting/underfitting
   - Adjust model complexity

5. **Feature Importance**
   - Consider feature interactions
   - Validate importance scores
   - Use domain knowledge

## Common Issues and Solutions

1. **Overfitting**
   - Increase regularization
   - Reduce model complexity
   - Use more training data

2. **Underfitting**
   - Increase model complexity
   - Add more features
   - Reduce regularization

3. **Computational Cost**
   - Use efficient search methods
   - Parallelize computations
   - Reduce parameter space

4. **Unbalanced Data**
   - Use stratified sampling
   - Consider class weights
   - Use appropriate metrics

## Examples

### 1. Classification Task

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_classification(n_samples=1000, n_features=20)

# Initialize evaluator
model = RandomForestClassifier()
evaluator = ModelEvaluator(model, task_type='classification')

# Evaluate model
metrics = evaluator.evaluate_performance(X, y)
evaluator.plot_confusion_matrix(y, model.predict(X))
```

### 2. Regression Task

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# Generate data
X, y = make_regression(n_samples=1000, n_features=20)

# Initialize evaluator
model = RandomForestRegressor()
evaluator = ModelEvaluator(model, task_type='regression')

# Evaluate model
metrics = evaluator.evaluate_performance(X, y)
evaluator.plot_learning_curve(X, y)
```

### 3. Feature Importance Analysis

```python
# Plot feature importance
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
evaluator.plot_feature_importance(feature_names)
```

## Contributing

Feel free to contribute to this module by:
1. Adding new evaluation metrics
2. Improving existing functionality
3. Adding more examples
4. Fixing bugs
5. Improving documentation

## License

This module is part of the ML Notes project and is available under the MIT License. 