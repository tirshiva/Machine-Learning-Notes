# Machine Learning Algorithms - Interview Preparation Guide

## Table of Contents
1. [Supervised Learning](#supervised-learning)
2. [Unsupervised Learning](#unsupervised-learning)
3. [Reinforcement Learning](#reinforcement-learning)
4. [Model Evaluation Metrics](#model-evaluation-metrics)
5. [ML Ops (Machine Learning Operations)](#ml-ops-machine-learning-operations)
6. [Common Interview Questions](#common-interview-questions)

## Supervised Learning

### 1. Linear Regression
- **Type**: Regression
- **Key Concepts**:
  - Fits a linear equation to observed data
  - Minimizes the sum of squared residuals
  - Assumes linear relationship between features and target
- **Use Cases**:
  - House price prediction
  - Sales forecasting
  - Risk assessment
- **Advantages**:
  - Simple to implement
  - Easy to interpret
  - Computationally efficient
- **Disadvantages**:
  - Assumes linearity
  - Sensitive to outliers
  - Cannot handle complex relationships

### 2. Logistic Regression
- **Type**: Classification
- **Key Concepts**:
  - Uses sigmoid function to output probabilities
  - Binary classification (can be extended to multi-class)
  - Maximum likelihood estimation
- **Use Cases**:
  - Spam detection
  - Credit scoring
  - Disease diagnosis
- **Advantages**:
  - Probabilistic interpretation
  - Less prone to overfitting
  - Works well with small datasets
- **Disadvantages**:
  - Assumes linear decision boundary
  - Cannot handle complex patterns
  - Requires feature scaling

### 3. Decision Trees
- **Type**: Classification/Regression
- **Key Concepts**:
  - Tree-like structure of decisions
  - Information gain/Gini impurity for splitting
  - Recursive binary splitting
- **Use Cases**:
  - Customer segmentation
  - Fraud detection
  - Medical diagnosis
- **Advantages**:
  - Easy to understand and interpret
  - Can handle both numerical and categorical data
  - Requires little data preprocessing
- **Disadvantages**:
  - Prone to overfitting
  - Can be unstable
  - May create biased trees

### 4. Random Forest
- **Type**: Classification/Regression
- **Key Concepts**:
  - Ensemble of decision trees
  - Bagging (Bootstrap Aggregating)
  - Feature randomness
- **Use Cases**:
  - Credit card fraud detection
  - Stock market prediction
  - Customer churn prediction
- **Advantages**:
  - Reduces overfitting
  - Handles high dimensionality
  - Robust to outliers
- **Disadvantages**:
  - Computationally expensive
  - Less interpretable
  - Requires more memory

### 5. Support Vector Machines (SVM)
- **Type**: Classification/Regression
- **Key Concepts**:
  - Maximum margin classifier
  - Kernel trick for non-linear data
  - Support vectors
- **Use Cases**:
  - Text classification
  - Image classification
  - Bioinformatics
- **Advantages**:
  - Effective in high dimensions
  - Memory efficient
  - Versatile (different kernels)
- **Disadvantages**:
  - Sensitive to noise
  - Computationally intensive
  - Requires careful tuning

## Unsupervised Learning

### 1. K-Means Clustering
- **Type**: Clustering
- **Key Concepts**:
  - Partitioning method
  - Minimizes within-cluster variance
  - Requires predefined number of clusters
- **Use Cases**:
  - Customer segmentation
  - Image compression
  - Document clustering
- **Advantages**:
  - Simple to implement
  - Scales well to large datasets
  - Guarantees convergence
- **Disadvantages**:
  - Requires number of clusters
  - Sensitive to outliers
  - Assumes spherical clusters

### 2. Hierarchical Clustering
- **Type**: Clustering
- **Key Concepts**:
  - Agglomerative or divisive approach
  - Creates cluster hierarchy
  - Dendrogram visualization
- **Use Cases**:
  - Taxonomy creation
  - Social network analysis
  - Market research
- **Advantages**:
  - No need to specify clusters
  - Easy to interpret
  - Works with any distance metric
- **Disadvantages**:
  - Computationally expensive
  - Sensitive to noise
  - Cannot handle large datasets

### 3. Principal Component Analysis (PCA)
- **Type**: Dimensionality Reduction
- **Key Concepts**:
  - Linear dimensionality reduction
  - Maximizes variance
  - Orthogonal transformation
- **Use Cases**:
  - Feature extraction
  - Data compression
  - Visualization
- **Advantages**:
  - Reduces noise
  - Speeds up learning
  - Removes correlation
- **Disadvantages**:
  - May lose important information
  - Assumes linear relationships
  - Sensitive to scaling

## Model Evaluation Metrics

### Classification Metrics

#### 1. Confusion Matrix
- **Components**:
  - True Positives (TP): Correctly predicted positive cases
  - True Negatives (TN): Correctly predicted negative cases
  - False Positives (FP): Incorrectly predicted positive cases
  - False Negatives (FN): Incorrectly predicted negative cases
- **Use Cases**:
  - Binary classification evaluation
  - Multi-class classification (one-vs-all approach)
- **Interpretation**:
  - Diagonal elements show correct predictions
  - Off-diagonal elements show errors

#### 2. Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Range**: 0 to 1 (higher is better)
- **Use Cases**:
  - Balanced datasets
  - When all classes are equally important
- **Limitations**:
  - Misleading for imbalanced datasets
  - Doesn't consider class distribution

#### 3. Precision
- **Formula**: TP / (TP + FP)
- **Interpretation**: Proportion of positive predictions that are actually positive
- **Use Cases**:
  - When cost of false positives is high
  - Spam detection
  - Fraud detection
- **Trade-off**: Higher precision often means lower recall

#### 4. Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Interpretation**: Proportion of actual positives correctly identified
- **Use Cases**:
  - Medical diagnosis
  - When missing positive cases is costly
- **Trade-off**: Higher recall often means lower precision

#### 5. F1 Score
- **Formula**: 2 * (Precision * Recall) / (Precision + Recall)
- **Interpretation**: Harmonic mean of precision and recall
- **Use Cases**:
  - When both precision and recall are important
  - Imbalanced datasets
- **Advantages**: Balances precision and recall

#### 6. ROC Curve and AUC
- **ROC Curve**: Plots True Positive Rate vs False Positive Rate
- **AUC**: Area Under the ROC Curve
- **Interpretation**:
  - AUC = 0.5: Random classifier
  - AUC = 1.0: Perfect classifier
  - Higher AUC indicates better model
- **Use Cases**:
  - Model comparison
  - Threshold selection
  - Overall model performance assessment

#### 7. Log Loss
- **Formula**: -1/N * Σ(y_i * log(p_i) + (1-y_i) * log(1-p_i))
- **Interpretation**: Measures accuracy of probabilistic predictions
- **Use Cases**:
  - When probability estimates are important
  - Multi-class classification
- **Advantages**: Penalizes confident wrong predictions

### Regression Metrics

#### 1. Mean Squared Error (MSE)
- **Formula**: 1/n * Σ(y_i - ŷ_i)²
- **Interpretation**: Average squared difference between predictions and actual values
- **Use Cases**:
  - General regression evaluation
  - When larger errors should be penalized more
- **Limitations**: Not in same unit as target variable

#### 2. Root Mean Squared Error (RMSE)
- **Formula**: √(1/n * Σ(y_i - ŷ_i)²)
- **Interpretation**: Square root of MSE, in same unit as target variable
- **Use Cases**:
  - When interpretability in original units is important
  - When larger errors should be penalized more
- **Advantages**: More interpretable than MSE

#### 3. Mean Absolute Error (MAE)
- **Formula**: 1/n * Σ|y_i - ŷ_i|
- **Interpretation**: Average absolute difference between predictions and actual values
- **Use Cases**:
  - When all errors should be weighted equally
  - When outliers are present
- **Advantages**: More robust to outliers than MSE

#### 4. R² (Coefficient of Determination)
- **Formula**: 1 - (SS_res / SS_tot)
- **Interpretation**: Proportion of variance in dependent variable predictable from independent variable(s)
- **Range**: -∞ to 1 (higher is better)
- **Use Cases**:
  - Overall model fit assessment
  - Model comparison
- **Limitations**: Can be misleading with non-linear relationships

#### 5. Adjusted R²
- **Formula**: 1 - (1-R²)(n-1)/(n-p-1)
- **Interpretation**: R² adjusted for number of predictors
- **Use Cases**:
  - When comparing models with different numbers of predictors
  - To prevent overfitting
- **Advantages**: Penalizes adding unnecessary predictors

#### 6. Mean Absolute Percentage Error (MAPE)
- **Formula**: 100/n * Σ|(y_i - ŷ_i)/y_i|
- **Interpretation**: Average percentage difference between predictions and actual values
- **Use Cases**:
  - When relative errors are important
  - Comparing models across different scales
- **Limitations**: Undefined when actual values are zero

### Model Evaluation Best Practices

#### 1. Cross-Validation
- **Types**:
  - K-Fold Cross-Validation
  - Stratified K-Fold
  - Leave-One-Out Cross-Validation
  - Time Series Cross-Validation
- **Benefits**:
  - More reliable performance estimate
  - Better use of limited data
  - Reduces overfitting

#### 2. Hyperparameter Tuning
- **Methods**:
  - Grid Search
  - Random Search
  - Bayesian Optimization
- **Best Practices**:
  - Use cross-validation
  - Define appropriate parameter ranges
  - Consider computational cost

#### 3. Learning Curves
- **Purpose**: Diagnose bias and variance
- **Interpretation**:
  - Gap between training and validation curves
  - Convergence behavior
- **Use Cases**:
  - Model selection
  - Identifying overfitting/underfitting

#### 4. Residual Analysis
- **For Regression**:
  - Plot residuals vs predicted values
  - Check for patterns
  - Verify normality
- **For Classification**:
  - Analyze misclassified examples
  - Look for patterns in errors

## ML Ops (Machine Learning Operations)

### 1. Version Control and Experiment Tracking
#### Tools and Techniques
- **MLflow**
  - Experiment tracking
  - Model versioning
  - Model registry
  - Deployment management
- **DVC (Data Version Control)**
  - Data versioning
  - Pipeline management
  - Experiment tracking
- **Weights & Biases**
  - Experiment tracking
  - Model visualization
  - Dataset versioning
- **Neptune.ai**
  - Experiment tracking
  - Model metadata management
  - Team collaboration

#### Best Practices
- Track all experiments with parameters
- Version control for:
  - Code
  - Data
  - Models
  - Configurations
- Document experiment results
- Maintain reproducibility

### 2. Model Deployment
#### Containerization
- **Docker**
  - Package models and dependencies
  - Ensure consistency across environments
  - Easy deployment and scaling
- **Kubernetes**
  - Container orchestration
  - Auto-scaling
  - Load balancing
  - High availability

#### Model Serving
- **TensorFlow Serving**
  - High-performance serving
  - Model versioning
  - A/B testing
- **TorchServe**
  - PyTorch model serving
  - Model versioning
  - REST API endpoints
- **Seldon Core**
  - Model serving on Kubernetes
  - Advanced deployment patterns
  - Monitoring and explainability

### 3. Model Monitoring
#### Performance Monitoring
- **Data Drift Detection**
  - Feature drift
  - Target drift
  - Concept drift
- **Model Performance Metrics**
  - Prediction latency
  - Throughput
  - Error rates
  - Resource utilization

#### Tools
- **Evidently**
  - Data quality monitoring
  - Model performance analysis
  - Data drift detection
- **Prometheus**
  - Metrics collection
  - Alerting
  - Time series data
- **Grafana**
  - Visualization
  - Dashboard creation
  - Alert management

### 4. CI/CD for ML
#### Continuous Integration
- **Testing**
  - Unit tests
  - Integration tests
  - Model validation tests
  - Data validation tests
- **Code Quality**
  - Linting
  - Type checking
  - Code coverage

#### Continuous Deployment
- **Automated Pipelines**
  - Model training
  - Model evaluation
  - Model deployment
- **Tools**
  - GitHub Actions
  - GitLab CI
  - Jenkins
  - Azure DevOps

### 5. Feature Store
#### Purpose
- Centralized feature management
- Feature reuse
- Consistent feature computation
- Real-time feature serving

#### Tools
- **Feast**
  - Feature definition
  - Feature serving
  - Offline/online feature storage
- **Hopsworks**
  - Feature store
  - Model registry
  - Experiment tracking
- **Tecton**
  - Feature platform
  - Real-time feature serving
  - Feature monitoring

### 6. Model Registry
#### Features
- Model versioning
- Model metadata
- Model lineage
- Model stage management

#### Tools
- **MLflow Model Registry**
  - Version control
  - Stage transitions
  - Model annotations
- **Azure ML Model Registry**
  - Model versioning
  - Model deployment
  - Model monitoring

### 7. Data Pipeline Management
#### Components
- **Data Ingestion**
  - Batch processing
  - Stream processing
  - Data validation
- **Data Transformation**
  - Feature engineering
  - Data cleaning
  - Data normalization
- **Data Storage**
  - Raw data
  - Processed data
  - Feature store

#### Tools
- **Apache Airflow**
  - Workflow orchestration
  - Task scheduling
  - Pipeline monitoring
- **Kubeflow**
  - ML pipeline orchestration
  - Experiment tracking
  - Model serving
- **Prefect**
  - Workflow management
  - Task scheduling
  - Pipeline monitoring

### 8. Security and Compliance
#### Security Measures
- **Model Security**
  - Model encryption
  - Access control
  - Secure serving
- **Data Security**
  - Data encryption
  - Access control
  - Data masking
- **Infrastructure Security**
  - Network security
  - Container security
  - API security

#### Compliance
- **Data Privacy**
  - GDPR compliance
  - Data anonymization
  - Privacy-preserving ML
- **Model Governance**
  - Model documentation
  - Audit trails
  - Compliance reporting

### 9. Cost Optimization
#### Strategies
- **Resource Optimization**
  - Auto-scaling
  - Resource allocation
  - Cost monitoring
- **Model Optimization**
  - Model quantization
  - Model pruning
  - Batch processing
- **Infrastructure Optimization**
  - Cloud cost management
  - Resource scheduling
  - Spot instances

#### Tools
- **Cloud Cost Management**
  - AWS Cost Explorer
  - Azure Cost Management
  - Google Cloud Billing
- **Resource Monitoring**
  - CloudWatch
  - Azure Monitor
  - Google Cloud Monitoring

### 10. Best Practices
#### Development
- Use version control for all components
- Implement automated testing
- Document all processes
- Follow coding standards

#### Deployment
- Use containerization
- Implement CI/CD pipelines
- Monitor model performance
- Plan for rollbacks

#### Maintenance
- Regular model retraining
- Performance monitoring
- Security updates
- Cost optimization

#### Team Collaboration
- Clear documentation
- Standardized processes
- Regular reviews
- Knowledge sharing

## Common Interview Questions

### Technical Questions
1. **What is the difference between supervised and unsupervised learning?**
   - Supervised: Uses labeled data, learns mapping from input to output
   - Unsupervised: Uses unlabeled data, finds patterns and structures

2. **Explain the bias-variance tradeoff**
   - Bias: Error from assumptions in the learning algorithm
   - Variance: Error from sensitivity to small fluctuations in training data
   - Tradeoff: Increasing model complexity reduces bias but increases variance

3. **What is cross-validation and why is it important?**
   - Technique to assess model performance
   - Helps prevent overfitting
   - Provides more reliable performance estimate

4. **How do you handle missing data?**
   - Deletion
   - Imputation (mean, median, mode)
   - Advanced methods (KNN, regression)

5. **Explain regularization and its types**
   - L1 (Lasso): Adds absolute value of coefficients
   - L2 (Ridge): Adds squared value of coefficients
   - Elastic Net: Combines L1 and L2

### Practical Questions
1. **How would you approach a new ML problem?**
   - Data collection and understanding
   - Data preprocessing
   - Feature engineering
   - Model selection
   - Training and evaluation
   - Deployment and monitoring

2. **How do you prevent overfitting?**
   - Cross-validation
   - Regularization
   - Early stopping
   - Feature selection
   - More training data

3. **How do you handle imbalanced datasets?**
   - Resampling (oversampling/undersampling)
   - Synthetic data generation (SMOTE)
   - Class weights
   - Anomaly detection approaches

4. **What's your approach to feature selection?**
   - Domain knowledge
   - Statistical methods
   - Model-based selection
   - Dimensionality reduction

5. **How do you evaluate model performance in production?**
   - A/B testing
   - Monitoring metrics
   - Feedback loops
   - Regular retraining 