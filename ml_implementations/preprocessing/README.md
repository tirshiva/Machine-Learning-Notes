# Data Preprocessing Module

This module provides comprehensive data preprocessing functionality for machine learning projects. It includes various techniques for handling different types of data and preparing it for model training.

## Features

### 1. Data Cleaning and Imputation
- Handling missing values using different strategies:
  - Mean/Median/Mode imputation
  - KNN imputation
  - Most frequent value imputation
- Automatic detection of column types
- Handling outliers

### 2. Feature Scaling and Normalization
- Standard scaling (z-score normalization)
- Min-Max scaling
- Robust scaling
- Automatic scaling of numerical features

### 3. Categorical Variable Handling
- One-hot encoding
- Label encoding
- Handling unknown categories
- Automatic detection of categorical columns

### 4. Text Preprocessing
- Text cleaning
- Tokenization
- Stop word removal
- Lemmatization
- Special character removal
- Case normalization

### 5. Time Series Feature Engineering
- Date component extraction
- Cyclical feature encoding
- Time-based aggregations
- Lag features
- Rolling statistics

### 6. Feature Selection
- Univariate feature selection
- Recursive feature elimination
- Feature importance based selection
- Automatic feature selection

## Usage

### Basic Usage

```python
from ml_implementations.preprocessing.data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    numerical_columns=['age', 'income'],
    categorical_columns=['gender', 'education'],
    text_columns=['review'],
    datetime_columns=['purchase_date'],
    imputation_strategy='mean',
    scaling_method='standard'
)

# Preprocess data
processed_df = preprocessor.fit_transform(df, target)
```

### Automatic Column Detection

```python
# Initialize preprocessor without specifying columns
preprocessor = DataPreprocessor()

# The preprocessor will automatically detect column types
processed_df = preprocessor.fit_transform(df, target)
```

### Feature Selection

```python
# Initialize preprocessor with feature selection
preprocessor = DataPreprocessor(
    feature_selection_method='kbest',
    n_features_to_select=5
)

# Preprocess data with feature selection
processed_df = preprocessor.fit_transform(df, target)
```

## Best Practices

1. **Data Quality**
   - Always check for missing values before preprocessing
   - Handle outliers appropriately
   - Validate data types and ranges

2. **Feature Engineering**
   - Create domain-specific features
   - Consider feature interactions
   - Use appropriate scaling methods

3. **Categorical Variables**
   - Use one-hot encoding for nominal variables
   - Use label encoding for ordinal variables
   - Handle unknown categories

4. **Text Data**
   - Remove irrelevant information
   - Use appropriate tokenization
   - Consider using advanced techniques like word embeddings

5. **Time Series Data**
   - Extract relevant temporal features
   - Handle seasonality and trends
   - Consider using cyclical encoding

## Common Issues and Solutions

1. **Memory Issues**
   - Use sparse matrices for one-hot encoding
   - Process data in batches
   - Remove unnecessary columns

2. **Performance Issues**
   - Use efficient data structures
   - Optimize preprocessing steps
   - Consider parallel processing

3. **Data Leakage**
   - Fit transformers on training data only
   - Transform test data using fitted transformers
   - Avoid using future information

## Examples

### 1. Handling Missing Values

```python
# Initialize preprocessor with KNN imputation
preprocessor = DataPreprocessor(
    imputation_strategy='knn'
)

# Preprocess data
processed_df = preprocessor.fit_transform(df)
```

### 2. Text Preprocessing

```python
# Initialize preprocessor with text columns
preprocessor = DataPreprocessor(
    text_columns=['review', 'description']
)

# Preprocess data
processed_df = preprocessor.fit_transform(df)
```

### 3. Time Series Feature Engineering

```python
# Initialize preprocessor with datetime columns
preprocessor = DataPreprocessor(
    datetime_columns=['timestamp', 'date']
)

# Preprocess data
processed_df = preprocessor.fit_transform(df)
```

## Contributing

Feel free to contribute to this module by:
1. Adding new preprocessing techniques
2. Improving existing functionality
3. Adding more examples
4. Fixing bugs
5. Improving documentation

## License

This module is part of the ML Notes project and is available under the MIT License. 