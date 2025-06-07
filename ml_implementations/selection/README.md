# Model Selection Module

This module provides comprehensive model selection functionality for machine learning projects. It includes various techniques for comparing models, creating ensembles, and optimizing model performance.

## Features

### 1. Model Comparison
- Cross-validation based comparison
- Multiple scoring metrics
- Statistical significance testing
- Visualization of results
- Performance metrics analysis

### 2. Ensemble Methods
- Voting ensembles
- Averaging ensembles
- Weighted ensembles
- Custom ensemble creation
- Ensemble evaluation

### 3. Model Stacking
- Cross-validation based stacking
- Meta-model selection
- Probability-based stacking
- Custom stacking configurations
- Stacking evaluation

### 4. Model Blending
- Validation set based blending
- Meta-model selection
- Probability-based blending
- Custom blending configurations
- Blending evaluation

## Usage

### Basic Usage

```python
from ml_implementations.selection.model_selection import ModelSelector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Initialize models
models = {
    'rf': RandomForestClassifier(),
    'gb': GradientBoostingClassifier(),
    'svm': SVC(probability=True)
}

# Initialize model selector
selector = ModelSelector(models, task_type='classification')

# Compare models
comparison_results = selector.compare_models(X, y)
print("Comparison Results:", comparison_results)

# Plot comparison
selector.plot_comparison()
```

### Creating Ensembles

```python
# Create voting ensemble
selector.create_ensemble(method='voting')
ensemble_metrics = selector.evaluate_ensemble(X, y)
print("Ensemble Metrics:", ensemble_metrics)

# Create averaging ensemble
selector.create_ensemble(method='averaging')
averaging_metrics = selector.evaluate_ensemble(X, y)
print("Averaging Metrics:", averaging_metrics)
```

### Model Stacking

```python
from sklearn.linear_model import LogisticRegression

# Create stacking ensemble
meta_model = LogisticRegression()
selector.create_stacking_ensemble(meta_model)
stacking_metrics = selector.evaluate_ensemble(X, y)
print("Stacking Metrics:", stacking_metrics)
```

### Model Blending

```python
# Create blending ensemble
selector.create_blending_ensemble(meta_model)
blending_metrics = selector.evaluate_ensemble(X, y)
print("Blending Metrics:", blending_metrics)
```

## Best Practices

1. **Model Comparison**
   - Use appropriate cross-validation
   - Consider multiple metrics
   - Check statistical significance
   - Visualize results

2. **Ensemble Methods**
   - Use diverse base models
   - Consider model weights
   - Validate ensemble performance
   - Monitor computational cost

3. **Model Stacking**
   - Use appropriate meta-model
   - Consider probability outputs
   - Validate stacking performance
   - Monitor overfitting

4. **Model Blending**
   - Use appropriate validation size
   - Consider probability outputs
   - Validate blending performance
   - Monitor overfitting

## Common Issues and Solutions

1. **Overfitting**
   - Use appropriate cross-validation
   - Monitor validation performance
   - Regularize meta-models
   - Use diverse base models

2. **Computational Cost**
   - Use efficient base models
   - Parallelize computations
   - Reduce number of models
   - Use appropriate validation size

3. **Model Diversity**
   - Use different algorithms
   - Use different parameters
   - Use different features
   - Use different preprocessing

4. **Performance Tuning**
   - Tune base models
   - Tune meta-models
   - Tune ensemble weights
   - Tune validation size

## Examples

### 1. Classification Task

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Generate data
X, y = make_classification(n_samples=1000, n_features=20)

# Initialize models
models = {
    'rf': RandomForestClassifier(),
    'gb': GradientBoostingClassifier(),
    'svm': SVC(probability=True)
}

# Initialize selector
selector = ModelSelector(models, task_type='classification')

# Compare models
comparison_results = selector.compare_models(X, y)
selector.plot_comparison()

# Create ensembles
selector.create_ensemble(method='voting')
selector.create_stacking_ensemble(LogisticRegression())
selector.create_blending_ensemble(LogisticRegression())
```

### 2. Regression Task

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Generate data
X, y = make_regression(n_samples=1000, n_features=20)

# Initialize models
models = {
    'rf': RandomForestRegressor(),
    'gb': GradientBoostingRegressor(),
    'svr': SVR()
}

# Initialize selector
selector = ModelSelector(models, task_type='regression')

# Compare models
comparison_results = selector.compare_models(X, y)
selector.plot_comparison()

# Create ensembles
selector.create_ensemble(method='averaging')
selector.create_stacking_ensemble(LinearRegression())
selector.create_blending_ensemble(LinearRegression())
```

### 3. Custom Ensemble

```python
# Create custom ensemble with weights
weights = [0.4, 0.3, 0.3]
selector.create_ensemble(method='voting', weights=weights)
ensemble_metrics = selector.evaluate_ensemble(X, y)
print("Custom Ensemble Metrics:", ensemble_metrics)
```

## Contributing

Feel free to contribute to this module by:
1. Adding new ensemble methods
2. Improving existing functionality
3. Adding more examples
4. Fixing bugs
5. Improving documentation

## License

This module is part of the ML Notes project and is available under the MIT License. 