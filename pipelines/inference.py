"""
Inference Pipeline for PayPal Fraud Detection
===========================================

This module implements the inference pipeline that serves predictions
for real-time fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import pickle
import joblib
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class FraudDetectionInference:
    """Inference pipeline for real-time fraud detection."""
    
    def __init__(self, model_registry_path: str = "models/model_registry.pkl"):
        """
        Initialize the inference pipeline.
        
        Args:
            model_registry_path: Path to the model registry file
        """
        self.model_registry_path = model_registry_path
        self.feature_engineer = None
        self.scaler = None
        self.models = {}
        self.best_model_name = None
        self.feature_names = []
        self.decision_thresholds = {
            'LOCK_USER': 0.8,      # High confidence fraud
            'ALERT_AGENT': 0.3,    # Medium suspicion
            'APPROVE': 0.0         # Low risk
        }
        
        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'fraud_predictions': 0,
            'lock_actions': 0,
            'alert_actions': 0,
            'approve_actions': 0
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load models and components from registry."""
        try:
            logger.info(f"Loading models from {self.model_registry_path}")
            
            with open(self.model_registry_path, 'rb') as f:
                registry_info = pickle.load(f)
            
            # Load feature engineer
            self.feature_engineer = FeatureEngineer.load(registry_info['feature_engineer_path'])
            
            # Load scaler
            self.scaler = joblib.load(registry_info['scaler_path'])
            
            # Load models
            for model_name, model_path in registry_info['models'].items():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} from {model_path}")
            
            self.best_model_name = registry_info['best_model']
            logger.info(f"Best model: {self.best_model_name}")
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict_single_transaction(self, user_data: Dict, transaction_data: Dict, 
                                 model_name: Optional[str] = None) -> Dict[str, Union[float, str, Dict]]:
        """
        Predict fraud probability for a single transaction.
        
        Args:
            user_data: Dictionary with user information
            transaction_data: Dictionary with transaction information
            model_name: Optional specific model to use (defaults to best model)
            
        Returns:
            Dictionary with prediction results and recommended action
        """
        try:
            # Use best model if not specified
            if model_name is None:
                model_name = self.best_model_name
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # Convert to DataFrame format expected by feature engineer
            user_df = pd.DataFrame([user_data])
            transaction_df = pd.DataFrame([transaction_data])
            
            # Ensure required columns exist
            self._validate_input_data(user_df, transaction_df)
            
            # Engineer features
            features = self._create_features_for_inference(user_df, transaction_df)
            
            # Make prediction
            model = self.models[model_name]
            
            if model_name == 'logistic_regression':
                features_scaled = self.scaler.transform(features)
                fraud_probability = model.predict_proba(features_scaled)[0, 1]
            else:
                fraud_probability = model.predict_proba(features)[0, 1]
            
            # Determine action
            action = self._determine_action(fraud_probability)
            
            # Update stats
            self._update_prediction_stats(fraud_probability, action)
            
            # Prepare result
            result = {
                'fraud_probability': float(fraud_probability),
                'is_fraud_prediction': fraud_probability > 0.5,
                'confidence_level': self._get_confidence_level(fraud_probability),
                'recommended_action': action,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat(),
                'feature_contributions': self._get_feature_contributions(features, model, model_name),
                'risk_factors': self._identify_risk_factors(user_data, transaction_data, features)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return safe default in case of error
            return {
                'fraud_probability': 0.5,
                'is_fraud_prediction': True,
                'confidence_level': 'LOW',
                'recommended_action': 'ALERT_AGENT',
                'model_used': 'ERROR_FALLBACK',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def predict_batch(self, users_data: List[Dict], transactions_data: List[Dict],
                     model_name: Optional[str] = None) -> List[Dict]:
        """
        Predict fraud probability for a batch of transactions.
        
        Args:
            users_data: List of user data dictionaries
            transactions_data: List of transaction data dictionaries
            model_name: Optional specific model to use
            
        Returns:
            List of prediction results
        """
        try:
            if len(users_data) != len(transactions_data):
                raise ValueError("Users and transactions data must have same length")
            
            # Use best model if not specified
            if model_name is None:
                model_name = self.best_model_name
            
            # Convert to DataFrames
            users_df = pd.DataFrame(users_data)
            transactions_df = pd.DataFrame(transactions_data)
            
            # Validate input
            self._validate_input_data(users_df, transactions_df)
            
            # Engineer features
            features = self._create_features_for_inference(users_df, transactions_df)
            
            # Make predictions
            model = self.models[model_name]
            
            if model_name == 'logistic_regression':
                features_scaled = self.scaler.transform(features)
                fraud_probabilities = model.predict_proba(features_scaled)[:, 1]
            else:
                fraud_probabilities = model.predict_proba(features)[:, 1]
            
            # Process results
            results = []
            for i, (user_data, transaction_data, prob) in enumerate(
                zip(users_data, transactions_data, fraud_probabilities)
            ):
                action = self._determine_action(prob)
                self._update_prediction_stats(prob, action)
                
                result = {
                    'fraud_probability': float(prob),
                    'is_fraud_prediction': prob > 0.5,
                    'confidence_level': self._get_confidence_level(prob),
                    'recommended_action': action,
                    'model_used': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'transaction_index': i
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return safe defaults
            return [
                {
                    'fraud_probability': 0.5,
                    'is_fraud_prediction': True,
                    'confidence_level': 'LOW',
                    'recommended_action': 'ALERT_AGENT',
                    'model_used': 'ERROR_FALLBACK',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                for _ in range(len(users_data))
            ]
    
    async def predict_async(self, user_data: Dict, transaction_data: Dict,
                          model_name: Optional[str] = None) -> Dict:
        """
        Asynchronous prediction for high-throughput scenarios.
        
        Args:
            user_data: User information
            transaction_data: Transaction information
            model_name: Optional specific model to use
            
        Returns:
            Prediction results
        """
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                self.predict_single_transaction,
                user_data, transaction_data, model_name
            )
        
        return result
    
    def _validate_input_data(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame):
        """Validate input data format and required fields."""
        # Required user fields
        required_user_fields = ['id', 'has_email', 'country', 'birth_year', 'kyc']
        missing_user_fields = [field for field in required_user_fields 
                              if field not in users_df.columns]
        
        if missing_user_fields:
            logger.warning(f"Missing user fields: {missing_user_fields}")
            # Fill with defaults
            for field in missing_user_fields:
                if field == 'has_email':
                    users_df[field] = 1
                elif field == 'birth_year':
                    users_df[field] = 1985
                elif field == 'kyc':
                    users_df[field] = 'PASSED'
                else:
                    users_df[field] = 'UNKNOWN'
        
        # Required transaction fields
        required_txn_fields = ['user_id', 'amount_usd', 'currency', 'type']
        missing_txn_fields = [field for field in required_txn_fields 
                             if field not in transactions_df.columns]
        
        if missing_txn_fields:
            logger.warning(f"Missing transaction fields: {missing_txn_fields}")
            # Fill with defaults
            for field in missing_txn_fields:
                if field == 'amount_usd':
                    transactions_df[field] = 100
                elif field == 'currency':
                    transactions_df[field] = 'USD'
                elif field == 'type':
                    transactions_df[field] = 'CARD_PAYMENT'
                else:
                    transactions_df[field] = 'UNKNOWN'
    
    def _create_features_for_inference(self, users_df: pd.DataFrame, 
                                     transactions_df: pd.DataFrame) -> np.ndarray:
        """Create features for inference using the fitted feature engineer."""
        # Add current timestamp for time-based features
        current_time = datetime.now()
        transactions_df['created_date'] = current_time
        users_df['created_date'] = current_time - pd.Timedelta(days=30)  # Default account age
        
        # Transform user features
        user_features = self.feature_engineer.transform_user_features(users_df)
        
        # Transform transaction features
        transaction_features = self.feature_engineer.transform_transaction_features(transactions_df)
        
        # For inference, we need to simulate user transaction history
        # Create minimal user aggregation features with defaults
        user_txn_features = pd.DataFrame({
            'user_id': transactions_df['user_id'],
            'user_count_amount_usd': 5.0,  # Default transaction count
            'user_sum_amount_usd': 500.0,  # Default sum
            'user_mean_amount_usd': 100.0,  # Default mean
            'user_std_amount_usd': 50.0,   # Default std
            'user_min_amount_usd': 10.0,   # Default min
            'user_max_amount_usd': 200.0,  # Default max
            'user_sum_is_completed': 4.0,
            'user_sum_is_declined': 1.0,
            'user_sum_is_failed': 0.0,
            'user_sum_is_night_time': 1.0,
            'user_sum_is_weekend': 1.0,
            'user_mean_merchant_category_risk': 0.03,
            'user_mean_currency_risk_score': 0.03,
            'user_transaction_frequency': 0.7,
            'user_avg_daily_amount': 71.4,
            'user_decline_rate': 0.2,
            'user_night_transaction_rate': 0.2
        })
        
        # Merge all features
        final_features = transaction_features.copy()
        
        # Add user features
        final_features = final_features.merge(
            user_features, left_on='user_id', right_on='id', how='left', suffixes=('', '_user')
        )
        
        # Add user transaction aggregations
        final_features = final_features.merge(
            user_txn_features, on='user_id', how='left'
        )
        
        # Fill missing values
        final_features = final_features.fillna(0)
        
        # Select feature columns (exclude IDs)
        feature_columns = [col for col in final_features.columns 
                          if col not in ['id', 'user_id', 'id_user']]
        
        return final_features[feature_columns].values
    
    def _determine_action(self, fraud_probability: float) -> str:
        """Determine recommended action based on fraud probability."""
        if fraud_probability >= self.decision_thresholds['LOCK_USER']:
            return 'LOCK_USER'
        elif fraud_probability >= self.decision_thresholds['ALERT_AGENT']:
            return 'ALERT_AGENT'
        else:
            return 'APPROVE'
    
    def _get_confidence_level(self, fraud_probability: float) -> str:
        """Get confidence level for the prediction."""
        if fraud_probability >= 0.8 or fraud_probability <= 0.2:
            return 'HIGH'
        elif fraud_probability >= 0.6 or fraud_probability <= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_feature_contributions(self, features: np.ndarray, model, model_name: str) -> Dict:
        """Get feature contributions for interpretation."""
        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models, use feature importance as proxy
                importances = model.feature_importances_
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                # Get top contributing features
                top_indices = np.argsort(importances)[-5:][::-1]
                top_contributions = {
                    feature_names[i]: float(importances[i]) for i in top_indices
                }
                return top_contributions
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not compute feature contributions: {e}")
            return {}
    
    def _identify_risk_factors(self, user_data: Dict, transaction_data: Dict, 
                             features: np.ndarray) -> List[str]:
        """Identify specific risk factors for this transaction."""
        risk_factors = []
        
        # Check transaction amount
        amount = transaction_data.get('amount_usd', 0)
        if amount > 1000:
            risk_factors.append('HIGH_TRANSACTION_AMOUNT')
        
        # Check user KYC status
        kyc_status = user_data.get('kyc', 'UNKNOWN')
        if kyc_status in ['FAILED', 'PENDING', 'NONE']:
            risk_factors.append('INCOMPLETE_KYC')
        
        # Check failed sign-in attempts
        failed_attempts = user_data.get('failed_sign_in_attempts', 0)
        if failed_attempts > 0:
            risk_factors.append('PREVIOUS_FAILED_SIGNIN')
        
        # Check transaction type
        txn_type = transaction_data.get('type', '')
        if txn_type in ['ATM', 'P2P']:
            risk_factors.append('HIGH_RISK_TRANSACTION_TYPE')
        
        # Check merchant category
        merchant_category = transaction_data.get('merchant_category', '')
        high_risk_categories = ['atm', 'bar', 'casino']
        if merchant_category in high_risk_categories:
            risk_factors.append('HIGH_RISK_MERCHANT_CATEGORY')
        
        return risk_factors
    
    def _update_prediction_stats(self, fraud_probability: float, action: str):
        """Update prediction statistics."""
        self.prediction_stats['total_predictions'] += 1
        
        if fraud_probability > 0.5:
            self.prediction_stats['fraud_predictions'] += 1
        
        if action == 'LOCK_USER':
            self.prediction_stats['lock_actions'] += 1
        elif action == 'ALERT_AGENT':
            self.prediction_stats['alert_actions'] += 1
        else:
            self.prediction_stats['approve_actions'] += 1
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics."""
        stats = self.prediction_stats.copy()
        
        if stats['total_predictions'] > 0:
            stats['fraud_rate'] = stats['fraud_predictions'] / stats['total_predictions']
            stats['lock_rate'] = stats['lock_actions'] / stats['total_predictions']
            stats['alert_rate'] = stats['alert_actions'] / stats['total_predictions']
            stats['approve_rate'] = stats['approve_actions'] / stats['total_predictions']
        
        return stats
    
    def update_decision_thresholds(self, new_thresholds: Dict[str, float]):
        """Update decision thresholds for actions."""
        self.decision_thresholds.update(new_thresholds)
        logger.info(f"Updated decision thresholds: {self.decision_thresholds}")
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on the inference pipeline."""
        health_status = {
            'feature_engineer_loaded': self.feature_engineer is not None,
            'scaler_loaded': self.scaler is not None,
            'models_loaded': len(self.models) > 0,
            'best_model_available': self.best_model_name in self.models
        }
        
        health_status['overall_healthy'] = all(health_status.values())
        
        return health_status