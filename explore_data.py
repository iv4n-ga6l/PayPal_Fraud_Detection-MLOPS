"""
PayPal Fraud Detection - Data Exploration
=========================================

This script performs comprehensive exploration of the PayPal fraud detection dataset.
It analyzes user profiles and transaction data to understand patterns and features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Tuple, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class DataExplorer:
    """Comprehensive data exploration for PayPal fraud detection dataset."""
    
    def __init__(self, data_path: str = "data"):
        """Initialize the data explorer with data path."""
        self.data_path = Path(data_path)
        self.users_df = None
        self.transactions_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load users and transactions data."""
        logger.info("Loading datasets...")
        
        try:
            # Load users data
            users_file = self.data_path / "paypal_users.csv"
            self.users_df = pd.read_csv(users_file)
            logger.info(f"Users data loaded: {self.users_df.shape}")
            
            # Load transactions data (sample first to handle large file)
            transactions_file = self.data_path / "paypal_transactions.csv"
            # Read a sample first to understand structure
            self.transactions_df = pd.read_csv(transactions_file, nrows=10000)
            logger.info(f"Transactions sample loaded: {self.transactions_df.shape}")
            
            return self.users_df, self.transactions_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def basic_info(self, df: pd.DataFrame, name: str) -> Dict:
        """Get basic information about a dataframe."""
        logger.info(f"\n=== {name} Dataset Overview ===")
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage']:.2f} MB")
        print(f"Columns: {info['columns']}")
        print("\nData Types:")
        for col, dtype in info['dtypes'].items():
            print(f"  {col}: {dtype}")
        
        print("\nMissing Values:")
        for col, null_pct in info['null_percentages'].items():
            if null_pct > 0:
                print(f"  {col}: {info['null_counts'][col]} ({null_pct:.2f}%)")
        
        print("\nFirst few rows:")
        print(df.head())
        
        return info
    
    def explore_users_data(self) -> Dict:
        """Explore users dataset in detail."""
        if self.users_df is None:
            logger.error("Users data not loaded!")
            return {}
        
        logger.info("\n=== USERS DATA EXPLORATION ===")
        
        # Basic info
        users_info = self.basic_info(self.users_df, "Users")
        
        # User distribution analysis
        print("\n=== User Distribution Analysis ===")
        
        # Check for fraud labels
        if 'is_fraud' in self.users_df.columns:
            fraud_dist = self.users_df['is_fraud'].value_counts()
            fraud_pct = self.users_df['is_fraud'].value_counts(normalize=True) * 100
            print(f"\nFraud Distribution:")
            print(f"  Non-Fraud: {fraud_dist[0]} ({fraud_pct[0]:.2f}%)")
            print(f"  Fraud: {fraud_dist[1]} ({fraud_pct[1]:.2f}%)")
            
            # Plot fraud distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            self.users_df['is_fraud'].value_counts().plot(kind='bar', color=['green', 'red'])
            plt.title('Fraud vs Non-Fraud Users')
            plt.xlabel('Is Fraud')
            plt.ylabel('Count')
            plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
            
            plt.subplot(1, 2, 2)
            fraud_pct.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'])
            plt.title('Fraud Distribution (%)')
            
            plt.tight_layout()
            plt.show()
        
        # Categorical columns analysis
        categorical_cols = self.users_df.select_dtypes(include=['object']).columns
        print(f"\nCategorical Columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            if col != 'user_id':  # Skip ID columns
                print(f"\n{col} distribution:")
                value_counts = self.users_df[col].value_counts().head(10)
                print(value_counts)
        
        # Numerical columns analysis
        numerical_cols = self.users_df.select_dtypes(include=['int64', 'float64']).columns
        print(f"\nNumerical Columns: {list(numerical_cols)}")
        
        if len(numerical_cols) > 0:
            print("\nNumerical Statistics:")
            print(self.users_df[numerical_cols].describe())
        
        return users_info
    
    def explore_transactions_data(self) -> Dict:
        """Explore transactions dataset in detail."""
        if self.transactions_df is None:
            logger.error("Transactions data not loaded!")
            return {}
        
        logger.info("\n=== TRANSACTIONS DATA EXPLORATION ===")
        
        # Basic info
        transactions_info = self.basic_info(self.transactions_df, "Transactions")
        
        print("\n=== Transaction Analysis ===")
        
        # Check for fraud labels
        if 'is_fraud' in self.transactions_df.columns:
            fraud_dist = self.transactions_df['is_fraud'].value_counts()
            fraud_pct = self.transactions_df['is_fraud'].value_counts(normalize=True) * 100
            print(f"\nTransaction Fraud Distribution:")
            print(f"  Non-Fraud: {fraud_dist[0]} ({fraud_pct[0]:.2f}%)")
            print(f"  Fraud: {fraud_dist[1]} ({fraud_pct[1]:.2f}%)")
        
        # Amount analysis
        if 'amount' in self.transactions_df.columns:
            print(f"\nTransaction Amount Statistics:")
            print(self.transactions_df['amount'].describe())
            
            # Plot amount distribution
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            self.transactions_df['amount'].hist(bins=50, alpha=0.7)
            plt.title('Transaction Amount Distribution')
            plt.xlabel('Amount')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 3, 2)
            self.transactions_df['amount'].plot(kind='box')
            plt.title('Transaction Amount Box Plot')
            plt.ylabel('Amount')
            
            if 'is_fraud' in self.transactions_df.columns:
                plt.subplot(1, 3, 3)
                fraud_amounts = self.transactions_df[self.transactions_df['is_fraud'] == 1]['amount']
                non_fraud_amounts = self.transactions_df[self.transactions_df['is_fraud'] == 0]['amount']
                
                plt.hist([non_fraud_amounts, fraud_amounts], bins=30, alpha=0.7, 
                        label=['Non-Fraud', 'Fraud'], color=['green', 'red'])
                plt.title('Amount Distribution by Fraud Status')
                plt.xlabel('Amount')
                plt.ylabel('Frequency')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        # Categorical analysis
        categorical_cols = self.transactions_df.select_dtypes(include=['object']).columns
        print(f"\nCategorical Columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            if col not in ['transaction_id', 'user_id']:  # Skip ID columns
                print(f"\n{col} distribution:")
                value_counts = self.transactions_df[col].value_counts().head(10)
                print(value_counts)
        
        return transactions_info
    
    def analyze_fraud_patterns(self):
        """Analyze fraud patterns across both datasets."""
        if self.users_df is None or self.transactions_df is None:
            logger.error("Data not fully loaded!")
            return
        
        logger.info("\n=== FRAUD PATTERN ANALYSIS ===")
        
        # Merge datasets for joint analysis
        if 'user_id' in self.users_df.columns and 'user_id' in self.transactions_df.columns:
            merged_df = self.transactions_df.merge(self.users_df, on='user_id', how='left', suffixes=('_txn', '_user'))
            
            print(f"Merged dataset shape: {merged_df.shape}")
            
            # Analyze fraud patterns
            if 'is_fraud_txn' in merged_df.columns or 'is_fraud_user' in merged_df.columns:
                fraud_col = 'is_fraud_txn' if 'is_fraud_txn' in merged_df.columns else 'is_fraud_user'
                
                # Country analysis
                if 'country' in merged_df.columns:
                    country_fraud = merged_df.groupby('country')[fraud_col].agg(['count', 'sum', 'mean']).round(3)
                    country_fraud.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
                    country_fraud = country_fraud.sort_values('fraud_rate', ascending=False)
                    
                    print("\nTop 10 Countries by Fraud Rate:")
                    print(country_fraud.head(10))
                
                # Currency analysis
                if 'currency' in merged_df.columns:
                    currency_fraud = merged_df.groupby('currency')[fraud_col].agg(['count', 'sum', 'mean']).round(3)
                    currency_fraud.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
                    currency_fraud = currency_fraud.sort_values('fraud_rate', ascending=False)
                    
                    print("\nCurrency Fraud Analysis:")
                    print(currency_fraud.head(10))
        
    def generate_feature_insights(self) -> Dict:
        """Generate insights for feature engineering."""
        logger.info("\n=== FEATURE ENGINEERING INSIGHTS ===")
        
        insights = {
            'users_features': [],
            'transactions_features': [],
            'derived_features': []
        }
        
        if self.users_df is not None:
            print("\nUser Features for ML:")
            for col in self.users_df.columns:
                if col != 'user_id':
                    insights['users_features'].append(col)
                    print(f"  - {col}: {self.users_df[col].dtype}")
        
        if self.transactions_df is not None:
            print("\nTransaction Features for ML:")
            for col in self.transactions_df.columns:
                if col not in ['transaction_id', 'user_id']:
                    insights['transactions_features'].append(col)
                    print(f"  - {col}: {self.transactions_df[col].dtype}")
        
        print("\nSuggested Derived Features:")
        derived_features = [
            "transaction_frequency_last_24h",
            "avg_transaction_amount_last_7d",
            "user_account_age_days",
            "transaction_velocity",
            "unusual_time_patterns",
            "country_risk_score",
            "merchant_category_risk",
            "amount_deviation_from_user_avg"
        ]
        
        for feature in derived_features:
            insights['derived_features'].append(feature)
            print(f"  - {feature}")
        
        return insights
    
    def run_full_exploration(self) -> Dict:
        """Run complete data exploration pipeline."""
        logger.info("Starting comprehensive data exploration...")
        
        try:
            # Load data
            self.load_data()
            
            # Explore datasets
            users_info = self.explore_users_data()
            transactions_info = self.explore_transactions_data()
            
            # Analyze fraud patterns
            self.analyze_fraud_patterns()
            
            # Generate feature insights
            feature_insights = self.generate_feature_insights()
            
            # Summary
            exploration_summary = {
                'users_info': users_info,
                'transactions_info': transactions_info,
                'feature_insights': feature_insights,
                'recommendations': self._generate_recommendations()
            }
            
            logger.info("Data exploration completed successfully!")
            return exploration_summary
            
        except Exception as e:
            logger.error(f"Error during exploration: {e}")
            raise
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on exploration."""
        recommendations = [
            "1. Implement real-time feature engineering pipeline for transaction velocity metrics",
            "2. Create user behavior profiling with historical transaction patterns",
            "3. Develop country/region risk scoring based on fraud rates",
            "4. Implement time-based features (hour of day, day of week patterns)",
            "5. Create merchant category risk assessment",
            "6. Develop amount anomaly detection based on user's historical spending",
            "7. Implement account age and KYC status weighting in fraud scoring",
            "8. Create ensemble model combining rule-based and ML approaches"
        ]
        
        print("\n=== RECOMMENDATIONS ===")
        for rec in recommendations:
            print(rec)
        
        return recommendations


def main():
    """Main execution function."""
    print("PayPal Fraud Detection - Data Exploration")
    print("=" * 50)
    
    # Initialize explorer
    explorer = DataExplorer()
    
    # Run exploration
    results = explorer.run_full_exploration()
    
    print("\n" + "=" * 50)
    print("Exploration completed! Check the generated insights above.")


if __name__ == "__main__":
    main()
