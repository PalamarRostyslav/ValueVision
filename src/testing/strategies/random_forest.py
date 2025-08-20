"""
Random Forest optimization strategy for fine-tuning.

This module contains the FineTuningRandomForest class that optimizes
predictor performance using Random Forest regression with TF-IDF text features.
"""

import numpy as np
from typing import List, Callable, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

from ..fine_tuning import FineTuning
from src.data.models import Item


class FineTuningRandomForest(FineTuning):
    """
    Fine-tuning strategy that uses Random Forest regression with TF-IDF text features.
    
    This implementation:
    1. Extracts TF-IDF features from item descriptions
    2. Creates feature matrix using TF-IDF vectorization
    3. Trains Random Forest regressor on the features
    4. Returns optimized predictor function
    """
    
    def __init__(self, max_features: int = 1000, n_estimators: int = 100, 
            n_jobs: int = 8, random_state: int = 42):
        """
        Initialize Random Forest fine-tuning with TF-IDF.
        
        Args:
            max_features: Maximum number of TF-IDF features
            n_estimators: Number of trees in Random Forest
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        """
        super().__init__("Random Forest with TF-IDF")
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tfidf_vectorizer = None
        self.rf_model = None
        self.svr_model = None
    
    def prepare_documents(self, training_data: List[Item]) -> List[str]:
        """
        Extract and prepare documents from training data.
        
        Args:
            training_data: Training data to extract documents from
            
        Returns:
            List of document strings
        """
        documents = []
        for item in training_data:
            if hasattr(item, 'test_prompt'):
                doc = item.test_prompt()
            else:
                doc_parts = [
                    getattr(item, 'title', ''),
                    getattr(item, 'description', ''),
                ]
                doc = ' '.join(filter(None, doc_parts))
            documents.append(doc)
        return documents
    
    def train_tfidf(self, documents: List[str]) -> None:
        """
        Train TF-IDF vectorizer on documents.
        
        Args:
            documents: List of document strings
        """
        print("Training TF-IDF vectorizer...")
        np.random.seed(self.random_state)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        
        # Fit the vectorizer
        self.tfidf_vectorizer.fit(documents)
        
        print(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def create_feature_matrix(self, documents: List[str]) -> np.ndarray:
        """
        Create feature matrix from documents using TF-IDF.
        
        Args:
            documents: List of document strings
            
        Returns:
            Feature matrix (n_documents x max_features)
        """
        print("Creating TF-IDF feature matrix...")
        X_tfidf = self.tfidf_vectorizer.transform(documents)
        print(f"Feature matrix shape: {X_tfidf.shape}")
        return X_tfidf
    
    def train_models(self, X, y: np.ndarray) -> None:
        """
        Train both SVR and Random Forest models.
        
        Args:
            X: Feature matrix (sparse or dense)
            y: Target prices
        """
        print("Training LinearSVR model...")
        np.random.seed(self.random_state)
        self.svr_model = LinearSVR(random_state=self.random_state, max_iter=10000)
        self.svr_model.fit(X, y)
        
        print("Training Random Forest model...")
        self.rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.rf_model.fit(X, y)
    
    def evaluate_models(self, X_train, y_train: np.ndarray,
            X_val=None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate both models and return metrics.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {}
        
        # Training evaluation
        svr_train_pred = self.svr_model.predict(X_train)
        rf_train_pred = self.rf_model.predict(X_train)
        
        metrics['svr_train_mse'] = mean_squared_error(y_train, svr_train_pred)
        metrics['svr_train_r2'] = r2_score(y_train, svr_train_pred)
        metrics['rf_train_mse'] = mean_squared_error(y_train, rf_train_pred)
        metrics['rf_train_r2'] = r2_score(y_train, rf_train_pred)
        
        print(f"SVR Training - MSE: {metrics['svr_train_mse']:.2f}, R²: {metrics['svr_train_r2']:.3f}")
        print(f"Random Forest Training - MSE: {metrics['rf_train_mse']:.2f}, R²: {metrics['rf_train_r2']:.3f}")
        
        # Validation evaluation
        if X_val is not None and y_val is not None:
            svr_val_pred = self.svr_model.predict(X_val)
            rf_val_pred = self.rf_model.predict(X_val)
            
            metrics['svr_val_mse'] = mean_squared_error(y_val, svr_val_pred)
            metrics['svr_val_r2'] = r2_score(y_val, svr_val_pred)
            metrics['rf_val_mse'] = mean_squared_error(y_val, rf_val_pred)
            metrics['rf_val_r2'] = r2_score(y_val, rf_val_pred)
            
            print(f"SVR Validation - MSE: {metrics['svr_val_mse']:.2f}, R²: {metrics['svr_val_r2']:.3f}")
            print(f"Random Forest Validation - MSE: {metrics['rf_val_mse']:.2f}, R²: {metrics['rf_val_r2']:.3f}")
        
        return metrics
    
    def create_rf_predictor(self) -> Callable:
        """
        Create Random Forest-based predictor function.
        
        Returns:
            Random Forest predictor function
        """
        def random_forest_predictor(item):
            doc = item.test_prompt() if hasattr(item, 'test_prompt') else str(item.title)
            doc_vector = self.tfidf_vectorizer.transform([doc])
            prediction = self.rf_model.predict(doc_vector)[0]
            return max(1.0, float(prediction))
        
        random_forest_predictor.__name__ = "random_forest_tfidf_predictor"
        return random_forest_predictor
    
    def optimize_predictor(self, training_data: List[Item], validation_data: List[Item] = None) -> Callable:
        """
        Optimize predictor using Random Forest with TF-IDF features.
        
        Args:
            training_data: Training data for optimization
            validation_data: Validation data for evaluation
            
        Returns:
            Optimized Random Forest predictor function
        """
        print(f"\nStarting {self.name}...")
        print(f"Using {len(training_data)} training samples")
        
        # Prepare documents
        train_documents = self.prepare_documents(training_data)
        train_prices = np.array([item.price for item in training_data])
        
        # Train TF-IDF vectorizer
        self.train_tfidf(train_documents)
        
        # Create feature matrices
        X_train = self.create_feature_matrix(train_documents)
        
        X_val = None
        y_val = None
        if validation_data:
            print(f"Using {len(validation_data)} validation samples")
            val_documents = self.prepare_documents(validation_data)
            X_val = self.create_feature_matrix(val_documents)
            y_val = np.array([item.price for item in validation_data])
        
        # Train models
        self.train_models(X_train, train_prices)
        
        # Evaluate models
        metrics = self.evaluate_models(X_train, train_prices, X_val, y_val)
        
        # Create optimized predictor (Random Forest performs better typically)
        optimized_predictor = self.create_rf_predictor()
        
        # Store optimization info
        optimized_predictor.optimization_info = {
            'strategy': self.name,
            'training_r2': metrics['rf_train_r2'],
            'training_mse': metrics['rf_train_mse'],
            'validation_r2': metrics.get('rf_val_r2'),
            'validation_mse': metrics.get('rf_val_mse'),
            'svr_train_r2': metrics['svr_train_r2'],
            'model_type': 'RandomForest',
            'max_features': self.max_features,
            'n_estimators': self.n_estimators,
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_)
        }
        
        return optimized_predictor
