"""
Training Pipeline for PayPal Fraud Detection
==========================================

This module implements the training pipeline that trains ML models for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional
import logging
import pickle
import joblib
from datetime import datetime
from pathlib import Path

from .feature_engineering import FeatureEngineer, create_ml_features

logger = logging.getLogger(__name__)

class FraudDetectionTrainer:
    """Training pipeline for fraud detection models."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the trainer."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = None
        self.scaler = None
        self.models = {}
        self.model_metrics = {}
        self.feature_importance = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'max_iter': 1000,
                    'solver': 'liblinear'
                }
            }
        }
    
    def load_and_prepare_data(self, users_file: str, transactions_file: str, 
                            sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data.
        
        Args:
            users_file: Path to users CSV file
            transactions_file: Path to transactions CSV file
            sample_size: Optional sample size for large datasets
            
        Returns:
            Tuple of (users_df, transactions_df)
        """
        logger.info("Loading training data...")
        
        # Load users data
        users_df = pd.read_csv(users_file)
        logger.info(f"Loaded users data: {users_df.shape}")
        
        # Load transactions data
        if sample_size:
            transactions_df = pd.read_csv(transactions_file, nrows=sample_size)
            logger.info(f"Loaded transactions sample: {transactions_df.shape}")
        else:
            transactions_df = pd.read_csv(transactions_file)
            logger.info(f"Loaded full transactions data: {transactions_df.shape}")
        
        # Basic data quality checks
        logger.info("Performing data quality checks...")
        
        # Check for required columns
        required_user_cols = ['id', 'is_fraud']
        required_txn_cols = ['user_id']
        
        missing_user_cols = [col for col in required_user_cols if col not in users_df.columns]
        missing_txn_cols = [col for col in required_txn_cols if col not in transactions_df.columns]
        
        if missing_user_cols:
            raise ValueError(f"Missing required user columns: {missing_user_cols}")
        if missing_txn_cols:
            raise ValueError(f"Missing required transaction columns: {missing_txn_cols}")
        
        # Check fraud distribution
        fraud_rate = users_df['is_fraud'].mean()
        logger.info(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
        
        if fraud_rate < 0.001:
            logger.warning("Very low fraud rate detected. Consider class balancing techniques.")
        elif fraud_rate > 0.1:
            logger.warning("High fraud rate detected. Verify data quality.")
        
        return users_df, transactions_df
    
    def create_training_dataset(self, users_df: pd.DataFrame, 
                              transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create ML-ready training dataset with features and labels.
        
        Args:
            users_df: User data with fraud labels
            transactions_df: Transaction data
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Creating training dataset...")
        
        # Initialize and fit feature engineer
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.fit(users_df, transactions_df)
        
        # Create features
        features_df = create_ml_features(users_df, transactions_df, self.feature_engineer)
        
        # Create labels by merging transaction data with user fraud labels
        # Each transaction inherits the fraud label of its user
        transactions_with_labels = transactions_df.merge(
            users_df[['id', 'is_fraud']], 
            left_on='user_id', 
            right_on='id', 
            how='left'
        )
        
        labels = transactions_with_labels['is_fraud'].fillna(0).astype(int)
        
        # Ensure features and labels have same length
        min_length = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_length]
        labels = labels.iloc[:min_length]
        
        logger.info(f"Created training dataset: {features_df.shape}")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
        
        return features_df, labels
    
    def train_models(self, features_df: pd.DataFrame, labels: pd.Series, 
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train multiple fraud detection models.
        
        Args:
            features_df: Training features
            labels: Training labels
            test_size: Test set size for evaluation
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Initialize model
                model = config['model'](**config['params'])
                
                # Train model
                if model_name == 'logistic_regression':
                    # Use scaled features for logistic regression
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    # Use original features for tree-based models
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Evaluate model
                metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
                
                # Store results
                self.models[model_name] = model
                self.model_metrics[model_name] = metrics
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(
                        zip(features_df.columns, model.feature_importances_)
                    )
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{model_name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Cross-validation
        self._perform_cross_validation(X_train, y_train)
        
        # Select best model
        best_model_name = max(self.model_metrics.keys(), 
                             key=lambda x: self.model_metrics[x]['auc'])
        logger.info(f"Best model: {best_model_name}")
        
        results['best_model'] = best_model_name
        results['feature_names'] = list(features_df.columns)
        
        return results
    
    def _evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        }
        
        return metrics
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5):
        """Perform cross-validation for all models."""
        logger.info("Performing cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                self.model_metrics[model_name]['cv_auc_mean'] = scores.mean()
                self.model_metrics[model_name]['cv_auc_std'] = scores.std()
                
                logger.info(f"{model_name} CV AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
                
            except Exception as e:
                logger.warning(f"CV failed for {model_name}: {e}")
    
    def generate_model_reports(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generate comprehensive model reports."""
        logger.info("Generating model reports...")
        
        # Create reports directory
        reports_dir = self.models_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                # Make predictions
                if model_name == 'logistic_regression':
                    X_test_scaled = self.scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Generate plots
                self._plot_model_performance(
                    y_test, y_pred, y_pred_proba, model_name, 
                    save_path=reports_dir / f"{model_name}_performance.png"
                )
                
                # Feature importance plot
                if model_name in self.feature_importance:
                    self._plot_feature_importance(
                        self.feature_importance[model_name], model_name,
                        save_path=reports_dir / f"{model_name}_feature_importance.png"
                    )
                
            except Exception as e:
                logger.error(f"Error generating report for {model_name}: {e}")
    
    def _plot_model_performance(self, y_true, y_pred, y_pred_proba, model_name, save_path):
        """Plot model performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} Performance', fontsize=16)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            axes[0,1].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
            axes[0,1].plot([0, 1], [0, 1], 'k--')
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate')
            axes[0,1].set_title('ROC Curve')
            axes[0,1].legend()
        
        # Precision-Recall Curve
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            axes[1,0].plot(recall, precision)
            axes[1,0].set_xlabel('Recall')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].set_title('Precision-Recall Curve')
        
        # Prediction Distribution
        axes[1,1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Non-Fraud', density=True)
        axes[1,1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Fraud', density=True)
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Prediction Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, importance_dict, model_name, save_path, top_n=20):
        """Plot feature importance."""
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self):
        """Save trained models and components."""
        logger.info("Saving models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save feature engineer
        feature_engineer_path = self.models_dir / f"feature_engineer_{timestamp}.pkl"
        self.feature_engineer.save(str(feature_engineer_path))
        
        # Save scaler
        scaler_path = self.models_dir / f"scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = self.models_dir / f"{model_name}_{timestamp}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save metrics
        metrics_path = self.models_dir / f"metrics_{timestamp}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.model_metrics, f)
        
        # Save model registry info
        registry_info = {
            'timestamp': timestamp,
            'feature_engineer_path': str(feature_engineer_path),
            'scaler_path': str(scaler_path),
            'models': {
                name: str(self.models_dir / f"{name}_{timestamp}.pkl")
                for name in self.models.keys()
            },
            'metrics': self.model_metrics,
            'best_model': max(self.model_metrics.keys(), 
                            key=lambda x: self.model_metrics[x]['auc'])
        }
        
        registry_path = self.models_dir / "model_registry.pkl"
        with open(registry_path, 'wb') as f:
            pickle.dump(registry_info, f)
        
        logger.info("All models saved successfully")
        return registry_info
    
    def train_pipeline(self, users_file: str, transactions_file: str, 
                      sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            users_file: Path to users CSV file
            transactions_file: Path to transactions CSV file
            sample_size: Optional sample size for large datasets
            
        Returns:
            Dictionary with training results and model info
        """
        logger.info("Starting complete training pipeline...")
        
        try:
            # Load data
            users_df, transactions_df = self.load_and_prepare_data(
                users_file, transactions_file, sample_size
            )
            
            # Create training dataset
            features_df, labels = self.create_training_dataset(users_df, transactions_df)
            
            # Train models
            training_results = self.train_models(features_df, labels)
            
            # Generate reports
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, labels, test_size=0.2, random_state=42, stratify=labels
            )
            self.generate_model_reports(X_test, y_test)
            
            # Save models
            registry_info = self.save_models()
            
            # Combine results
            final_results = {
                'training_results': training_results,
                'registry_info': registry_info,
                'data_info': {
                    'users_shape': users_df.shape,
                    'transactions_shape': transactions_df.shape,
                    'features_shape': features_df.shape,
                    'fraud_rate': labels.mean()
                }
            }
            
            logger.info("Training pipeline completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise