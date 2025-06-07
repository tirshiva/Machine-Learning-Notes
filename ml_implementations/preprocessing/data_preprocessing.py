"""
Data Preprocessing Module

This module provides comprehensive data preprocessing functionality including:
- Data cleaning and imputation
- Feature scaling and normalization
- Feature selection
- Categorical variable handling
- Text preprocessing
- Time series feature engineering

Author: ML Notes
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A comprehensive data preprocessing class that handles various preprocessing tasks.
    """
    
    def __init__(
        self,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        imputation_strategy: str = 'mean',
        scaling_method: str = 'standard',
        feature_selection_method: Optional[str] = None,
        n_features_to_select: Optional[int] = None
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            numerical_columns (Optional[List[str]]): List of numerical column names
            categorical_columns (Optional[List[str]]): List of categorical column names
            text_columns (Optional[List[str]]): List of text column names
            datetime_columns (Optional[List[str]]): List of datetime column names
            target_column (Optional[str]): Name of the target column
            imputation_strategy (str): Strategy for handling missing values
            scaling_method (str): Method for scaling numerical features
            feature_selection_method (Optional[str]): Method for feature selection
            n_features_to_select (Optional[int]): Number of features to select
        """
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []
        self.text_columns = text_columns or []
        self.datetime_columns = datetime_columns or []
        self.target_column = target_column
        
        # Initialize transformers
        self._initialize_transformers(
            imputation_strategy,
            scaling_method,
            feature_selection_method,
            n_features_to_select
        )
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _initialize_transformers(
        self,
        imputation_strategy: str,
        scaling_method: str,
        feature_selection_method: Optional[str],
        n_features_to_select: Optional[int]
    ) -> None:
        """
        Initialize the transformers based on the specified methods.
        """
        # Imputation
        if imputation_strategy == 'knn':
            self.numerical_imputer = KNNImputer(n_neighbors=5)
        else:
            self.numerical_imputer = SimpleImputer(strategy=imputation_strategy)
        
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Scaling
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Feature selection
        self.feature_selector = None
        if feature_selection_method and n_features_to_select:
            if feature_selection_method == 'kbest':
                self.feature_selector = SelectKBest(
                    score_func=f_classif,
                    k=n_features_to_select
                )
            elif feature_selection_method == 'rfe':
                self.feature_selector = RFE(
                    estimator=RandomForestClassifier(),
                    n_features_to_select=n_features_to_select
                )
            elif feature_selection_method == 'from_model':
                self.feature_selector = SelectFromModel(
                    estimator=RandomForestClassifier(),
                    max_features=n_features_to_select
                )
    
    def detect_column_types(self, df: pd.DataFrame) -> None:
        """
        Automatically detect column types in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        for column in df.columns:
            if column == self.target_column:
                continue
                
            if pd.api.types.is_numeric_dtype(df[column]):
                self.numerical_columns.append(column)
            elif pd.api.types.is_datetime64_dtype(df[column]):
                self.datetime_columns.append(column)
            elif pd.api.types.is_object_dtype(df[column]):
                # Check if it's text data
                if df[column].str.len().mean() > 50:
                    self.text_columns.append(column)
                else:
                    self.categorical_columns.append(column)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df = df.copy()
        
        # Handle numerical columns
        if self.numerical_columns:
            df[self.numerical_columns] = self.numerical_imputer.fit_transform(
                df[self.numerical_columns]
            )
        
        # Handle categorical columns
        if self.categorical_columns:
            df[self.categorical_columns] = self.categorical_imputer.fit_transform(
                df[self.categorical_columns]
            )
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        if not self.numerical_columns:
            return df
        
        df = df.copy()
        df[self.numerical_columns] = self.scaler.fit_transform(
            df[self.numerical_columns]
        )
        return df
    
    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Encoding method ('onehot' or 'label')
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        if not self.categorical_columns:
            return df
        
        df = df.copy()
        
        if method == 'onehot':
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[self.categorical_columns])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(self.categorical_columns)
            )
            df = pd.concat([df.drop(self.categorical_columns, axis=1), encoded_df], axis=1)
        else:
            for column in self.categorical_columns:
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column])
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        return ' '.join(tokens)
    
    def handle_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle text features in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with processed text features
        """
        if not self.text_columns:
            return df
        
        df = df.copy()
        for column in self.text_columns:
            df[column] = df[column].apply(self.preprocess_text)
        
        return df
    
    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from datetime columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with extracted datetime features
        """
        if not self.datetime_columns:
            return df
        
        df = df.copy()
        for column in self.datetime_columns:
            df[f'{column}_year'] = df[column].dt.year
            df[f'{column}_month'] = df[column].dt.month
            df[f'{column}_day'] = df[column].dt.day
            df[f'{column}_dayofweek'] = df[column].dt.dayofweek
            df[f'{column}_hour'] = df[column].dt.hour
            
            # Add cyclical features
            df[f'{column}_month_sin'] = np.sin(2 * np.pi * df[column].dt.month / 12)
            df[f'{column}_month_cos'] = np.cos(2 * np.pi * df[column].dt.month / 12)
            df[f'{column}_day_sin'] = np.sin(2 * np.pi * df[column].dt.day / 31)
            df[f'{column}_day_cos'] = np.cos(2 * np.pi * df[column].dt.day / 31)
        
        return df
    
    def select_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Select features using the specified method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target (Optional[pd.Series]): Target variable
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        if not self.feature_selector or target is None:
            return df
        
        df = df.copy()
        selected_features = self.feature_selector.fit_transform(df, target)
        selected_indices = self.feature_selector.get_support()
        selected_columns = df.columns[selected_indices]
        
        return pd.DataFrame(selected_features, columns=selected_columns)
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target (Optional[pd.Series]): Target variable
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Detect column types if not specified
        if not any([
            self.numerical_columns,
            self.categorical_columns,
            self.text_columns,
            self.datetime_columns
        ]):
            self.detect_column_types(df)
        
        # Apply preprocessing steps
        df = self.handle_missing_values(df)
        df = self.scale_features(df)
        df = self.encode_categorical_features(df)
        df = self.handle_text_features(df)
        df = self.extract_datetime_features(df)
        df = self.select_features(df, target)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to new data using fitted transformers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        return self.fit_transform(df)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate numerical data
    numerical_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'score': np.random.normal(75, 15, n_samples)
    })
    
    # Generate categorical data
    categorical_data = pd.DataFrame({
        'gender': np.random.choice(['M', 'F'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    })
    
    # Generate text data
    text_data = pd.DataFrame({
        'review': [
            'This product is amazing! I love it.',
            'Not satisfied with the quality.',
            'Great value for money.',
            'Could be better.',
            'Excellent service and support.'
        ] * (n_samples // 5)
    })
    
    # Generate datetime data
    datetime_data = pd.DataFrame({
        'purchase_date': pd.date_range(
            start='2023-01-01',
            periods=n_samples,
            freq='D'
        )
    })
    
    # Combine all data
    df = pd.concat([
        numerical_data,
        categorical_data,
        text_data,
        datetime_data
    ], axis=1)
    
    # Add some missing values
    df.loc[df.sample(frac=0.1).index, 'age'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'income'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'gender'] = np.nan
    
    # Create target variable
    target = (df['age'] > 35).astype(int)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        numerical_columns=['age', 'income', 'score'],
        categorical_columns=['gender', 'education'],
        text_columns=['review'],
        datetime_columns=['purchase_date'],
        imputation_strategy='mean',
        scaling_method='standard',
        feature_selection_method='kbest',
        n_features_to_select=5
    )
    
    # Preprocess data
    processed_df = preprocessor.fit_transform(df, target)
    
    print("\nOriginal Data Shape:", df.shape)
    print("Processed Data Shape:", processed_df.shape)
    print("\nProcessed Data Head:")
    print(processed_df.head()) 