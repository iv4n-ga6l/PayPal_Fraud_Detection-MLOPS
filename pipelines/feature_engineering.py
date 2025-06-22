"""
Feature Engineering Pipeline for PayPal Fraud Detection
=====================================================

This module implements the feature engineering pipeline that transforms raw data
into ML-ready features for fraud detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging
import pickle

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline for fraud detection."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.country_risk_scores = {}
        self.merchant_risk_scores = {}
        self.currency_risk_scores = {}
        self.feature_stats = {}
        
    def fit(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data to learn risk scores and statistics.
        
        Args:
            users_df: User data with fraud labels
            transactions_df: Transaction data
            
        Returns:
            self: Fitted feature engineer
        """
        logger.info("Fitting feature engineer...")
        
        # Calculate country risk scores
        self._calculate_country_risk_scores(users_df)
        
        # Calculate merchant category risk scores
        self._calculate_merchant_risk_scores(transactions_df, users_df)
        
        # Calculate currency risk scores  
        self._calculate_currency_risk_scores(transactions_df, users_df)
        
        # Calculate feature statistics for normalization
        self._calculate_feature_statistics(users_df, transactions_df)
        
        logger.info("Feature engineer fitted successfully")
        return self
        
    def transform_user_features(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform user data into ML features.
        
        Args:
            users_df: Raw user data
            
        Returns:
            DataFrame with engineered user features
        """
        df = users_df.copy()
        
        # Basic features
        df['has_email'] = df['has_email'].astype(int)
        
        # Account age
        df['created_date'] = pd.to_datetime(df['created_date'])
        current_time = datetime.now()
        df['account_age_days'] = (current_time - df['created_date']).dt.days
        df['account_age_hours'] = (current_time - df['created_date']).dt.total_seconds() / 3600
        
        # Age features
        current_year = datetime.now().year
        df['user_age'] = current_year - df['birth_year']
        df['is_young_user'] = (df['user_age'] < 25).astype(int)
        df['is_senior_user'] = (df['user_age'] > 65).astype(int)
        
        # Risk scores
        df['country_risk_score'] = df['country'].map(self.country_risk_scores).fillna(0.03)  # Default risk
        
        # KYC features
        df['kyc_passed'] = (df['kyc'] == 'PASSED').astype(int)
        df['kyc_failed'] = (df['kyc'] == 'FAILED').astype(int)
        df['kyc_pending'] = (df['kyc'] == 'PENDING').astype(int)
        df['kyc_none'] = (df['kyc'] == 'NONE').astype(int)
        
        # Phone country features
        df['phone_country_matches_country'] = (df['phone_country'] == df['country']).astype(int)
        
        # Failed sign-in attempts features
        df['has_failed_signin'] = (df['failed_sign_in_attempts'] > 0).astype(int)
        df['multiple_failed_signin'] = (df['failed_sign_in_attempts'] > 1).astype(int)
        
        # Terms version features (recency)
        if 'terms_version' in df.columns:
            df['terms_version'] = df['terms_version'].fillna('2023-01-01')
            df['terms_version_date'] = pd.to_datetime(df['terms_version'])
            df['terms_version_age_days'] = (current_time - df['terms_version_date']).dt.days
        
        # Select final features
        feature_columns = [
            'has_email', 'account_age_days', 'account_age_hours', 'user_age',
            'is_young_user', 'is_senior_user', 'country_risk_score',
            'kyc_passed', 'kyc_failed', 'kyc_pending', 'kyc_none',
            'phone_country_matches_country', 'has_failed_signin', 'multiple_failed_signin',
            'failed_sign_in_attempts', 'terms_version_age_days'
        ]
        
        # Ensure all columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
                
        return df[['id'] + feature_columns]
    
    def transform_transaction_features(self, transactions_df: pd.DataFrame, 
                                     user_features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform transaction data into ML features.
        
        Args:
            transactions_df: Raw transaction data
            user_features_df: Pre-computed user features (optional)
            
        Returns:
            DataFrame with engineered transaction features
        """
        df = transactions_df.copy()
          # Basic transaction features
        df['amount_usd_log'] = np.log1p(df['amount_usd'])
        df['is_crypto'] = df['is_crypto'].astype(int) if 'is_crypto' in df.columns else 0
        
        # Transaction state features (handle missing state column)
        if 'state' in df.columns:
            df['is_completed'] = (df['state'] == 'COMPLETED').astype(int)
            df['is_declined'] = (df['state'] == 'DECLINED').astype(int)
            df['is_failed'] = (df['state'] == 'FAILED').astype(int)
            df['is_reverted'] = (df['state'] == 'REVERTED').astype(int)
        else:
            df['is_completed'] = 1  # Default to completed
            df['is_declined'] = 0
            df['is_failed'] = 0
            df['is_reverted'] = 0
        
        # Time features
        if 'created_date' in df.columns:
            df['created_date'] = pd.to_datetime(df['created_date'])
        else:
            # Use current time as default
            df['created_date'] = pd.to_datetime('now')
            
        df['hour_of_day'] = df['created_date'].dt.hour
        df['day_of_week'] = df['created_date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['is_night_time'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        # Risk scores
        df['merchant_category_risk'] = df['merchant_category'].map(self.merchant_risk_scores).fillna(0.03)
        df['currency_risk_score'] = df['currency'].map(self.currency_risk_scores).fillna(0.03)
        
        # Entry method features
        entry_methods = ['chip', 'cont', 'manu', 'misc', 'mags', 'mcon']
        for method in entry_methods:
            df[f'entry_method_{method}'] = (df['entry_method'] == method).astype(int)
        
        # Transaction type features
        transaction_types = ['CARD_PAYMENT', 'TOPUP', 'ATM', 'BANK_TRANSFER', 'P2P']
        for txn_type in transaction_types:
            df[f'type_{txn_type}'] = (df['type'] == txn_type).astype(int)
        
        # Source features
        sources = ['GAIA', 'HERA', 'INTERNAL', 'LETO', 'MINOS', 'CRONUS']
        for source in sources:
            df[f'source_{source}'] = (df['source'] == source).astype(int)
        
        # Currency features
        major_currencies = ['GBP', 'EUR', 'USD']
        for currency in major_currencies:
            df[f'currency_{currency}'] = (df['currency'] == currency).astype(int)
        
        # Amount features
        df['is_high_amount'] = (df['amount_usd'] > df['amount_usd'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['amount_usd'] < df['amount_usd'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['amount_usd'] % 100 == 0).astype(int)
        
        # Select feature columns
        feature_columns = [
            'amount_usd', 'amount_usd_log', 'is_crypto',
            'is_completed', 'is_declined', 'is_failed', 'is_reverted',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night_time', 'is_business_hours',
            'merchant_category_risk', 'currency_risk_score',
            'is_high_amount', 'is_low_amount', 'is_round_amount'
        ]
        
        # Add one-hot encoded features
        feature_columns.extend([f'entry_method_{method}' for method in entry_methods])
        feature_columns.extend([f'type_{txn_type}' for txn_type in transaction_types])
        feature_columns.extend([f'source_{source}' for source in sources])
        feature_columns.extend([f'currency_{currency}' for currency in major_currencies])
        
        # Ensure all columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
                
        return df[['id', 'user_id'] + feature_columns]
    
    def create_user_transaction_features(self, transactions_df: pd.DataFrame, 
                                       lookback_days: int = 7) -> pd.DataFrame:
        """
        Create user-level aggregated features from transaction history.
        
        Args:
            transactions_df: Transaction data with engineered features
            lookback_days: Number of days to look back for aggregations
              Returns:
            DataFrame with user-level transaction features
        """
        df = transactions_df.copy()
        
        # Ensure created_date exists
        if 'created_date' not in df.columns:
            df['created_date'] = pd.to_datetime('now')
        else:
            df['created_date'] = pd.to_datetime(df['created_date'])
        
        # Define cutoff date for lookback
        current_time = datetime.now()
        cutoff_date = current_time - timedelta(days=lookback_days)
          # Filter recent transactions
        recent_df = df[df['created_date'] >= cutoff_date]
        
        # Define aggregation columns with fallbacks
        agg_columns = {
            'amount_usd': ['count', 'sum', 'mean', 'std', 'min', 'max']
        }
        
        # Add optional columns if they exist
        optional_columns = {
            'is_completed': 'sum',
            'is_declined': 'sum', 
            'is_failed': 'sum',
            'is_night_time': 'sum',
            'is_weekend': 'sum',
            'merchant_category_risk': 'mean',
            'currency_risk_score': 'mean'
        }
        
        for col, agg_func in optional_columns.items():
            if col in recent_df.columns:
                agg_columns[col] = agg_func
        
        # Aggregate features per user
        user_agg_features = recent_df.groupby('user_id').agg(agg_columns).reset_index()
        
        # Flatten column names
        user_agg_features.columns = ['user_id'] + [
            f'user_{col[1]}_{col[0]}' if col[1] else f'user_{col[0]}'
            for col in user_agg_features.columns[1:]
        ]
        # Calculate additional features
        user_agg_features['user_transaction_frequency'] = user_agg_features['user_count_amount_usd'] / lookback_days
        user_agg_features['user_avg_daily_amount'] = user_agg_features['user_sum_amount_usd'] / lookback_days
        
        # Calculate decline rate if the column exists
        if 'user_sum_is_declined' in user_agg_features.columns:
            user_agg_features['user_decline_rate'] = (
                user_agg_features['user_sum_is_declined'] / user_agg_features['user_count_amount_usd']
            ).fillna(0)
        else:
            user_agg_features['user_decline_rate'] = 0.0
        
        # Calculate night transaction rate if the column exists
        if 'user_sum_is_night_time' in user_agg_features.columns:
            user_agg_features['user_night_transaction_rate'] = (
                user_agg_features['user_sum_is_night_time'] / user_agg_features['user_count_amount_usd']
            ).fillna(0)
        else:
            user_agg_features['user_night_transaction_rate'] = 0.0
        
        # Fill NaN values
        user_agg_features = user_agg_features.fillna(0)
        
        return user_agg_features
    
    def _calculate_country_risk_scores(self, users_df: pd.DataFrame):
        """Calculate fraud risk scores by country."""
        if 'is_fraud' in users_df.columns:
            country_stats = users_df.groupby('country')['is_fraud'].agg(['count', 'sum', 'mean'])
            # Only use countries with sufficient samples
            country_stats = country_stats[country_stats['count'] >= 10]
            self.country_risk_scores = country_stats['mean'].to_dict()
        else:
            # Default risk scores if no fraud labels
            self.country_risk_scores = {'GB': 0.02, 'FR': 0.03, 'US': 0.025}
    
    def _calculate_merchant_risk_scores(self, transactions_df: pd.DataFrame, users_df: pd.DataFrame):
        """Calculate fraud risk scores by merchant category."""
        if 'is_fraud' in users_df.columns:
            # Merge to get fraud labels for transactions
            merged_df = transactions_df.merge(users_df[['id', 'is_fraud']], 
                                            left_on='user_id', right_on='id', how='left')
            
            merchant_stats = merged_df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean'])
            merchant_stats = merchant_stats[merchant_stats['count'] >= 10]
            self.merchant_risk_scores = merchant_stats['mean'].to_dict()
        else:
            # Default risk scores
            self.merchant_risk_scores = {'atm': 0.05, 'bar': 0.04, 'restaurant': 0.02}
    
    def _calculate_currency_risk_scores(self, transactions_df: pd.DataFrame, users_df: pd.DataFrame):
        """Calculate fraud risk scores by currency."""
        if 'is_fraud' in users_df.columns:
            merged_df = transactions_df.merge(users_df[['id', 'is_fraud']], 
                                            left_on='user_id', right_on='id', how='left')
            
            currency_stats = merged_df.groupby('currency')['is_fraud'].agg(['count', 'sum', 'mean'])
            currency_stats = currency_stats[currency_stats['count'] >= 10]
            self.currency_risk_scores = currency_stats['mean'].to_dict()
        else:
            # Default risk scores
            self.currency_risk_scores = {'GBP': 0.025, 'EUR': 0.03, 'USD': 0.02}
    
    def _calculate_feature_statistics(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame):
        """Calculate feature statistics for normalization."""
        # User feature stats
        numeric_user_cols = users_df.select_dtypes(include=[np.number]).columns
        self.feature_stats['users'] = {
            'mean': users_df[numeric_user_cols].mean().to_dict(),
            'std': users_df[numeric_user_cols].std().to_dict()
        }
        
        # Transaction feature stats
        numeric_txn_cols = transactions_df.select_dtypes(include=[np.number]).columns
        self.feature_stats['transactions'] = {
            'mean': transactions_df[numeric_txn_cols].mean().to_dict(),
            'std': transactions_df[numeric_txn_cols].std().to_dict()
        }
    
    def save(self, filepath: str):
        """Save the fitted feature engineer."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Feature engineer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureEngineer':
        """Load a fitted feature engineer."""
        with open(filepath, 'rb') as f:
            feature_engineer = pickle.load(f)
        logger.info(f"Feature engineer loaded from {filepath}")
        return feature_engineer


def create_ml_features(users_df: pd.DataFrame, transactions_df: pd.DataFrame, 
                      feature_engineer: Optional[FeatureEngineer] = None) -> pd.DataFrame:
    """
    Create complete ML-ready features by combining user and transaction data.
    
    Args:
        users_df: User data
        transactions_df: Transaction data
        feature_engineer: Fitted feature engineer (if None, will create new one)
        
    Returns:
        DataFrame with complete ML features
    """
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(users_df, transactions_df)
    # Transform features
    user_features = feature_engineer.transform_user_features(users_df)
    transaction_features = feature_engineer.transform_transaction_features(transactions_df)
    
    # Create user-level aggregations using the transformed transaction features
    user_txn_features = feature_engineer.create_user_transaction_features(transaction_features)
    
    # Merge all features
    # Start with transaction features
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
    
    # Remove ID columns for ML
    feature_columns = [col for col in final_features.columns 
                      if col not in ['id', 'user_id', 'id_user']]
    
    return final_features[feature_columns]
