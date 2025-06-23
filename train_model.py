"""
PayPal Fraud Detection - Model Training Script
============================================

This script trains machine learning models for fraud detection using
the training pipeline. It handles data loading, feature engineering,
model training, evaluation, and model persistence.
"""

import logging
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from pipelines.training import FraudDetectionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train PayPal fraud detection models')
    
    parser.add_argument(
        '--users-file',
        type=str,
        default='data/paypal_users.csv',
        help='Path to users CSV file (default: data/paypal_users.csv)'
    )
    
    parser.add_argument(
        '--transactions-file',
        type=str,
        default='data/paypal_transactions.csv',
        help='Path to transactions CSV file (default: data/paypal_transactions.csv)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50000,
        help='Sample size for training (default: 50000, set to 0 for full dataset)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    
    parser.add_argument(
        '--quick-train',
        action='store_true',
        help='Quick training with smaller sample and fewer models'
    )
    
    args = parser.parse_args()
    
    if args.quick_train:
        args.sample_size = 10000
        logger.info("Quick training mode enabled - using smaller sample size")
    
    # Use full dataset if sample_size is 0
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    logger.info("=" * 60)
    logger.info("PayPal Fraud Detection - Model Training")
    logger.info("=" * 60)
    logger.info(f"Users file: {args.users_file}")
    logger.info(f"Transactions file: {args.transactions_file}")
    logger.info(f"Sample size: {sample_size or 'Full dataset'}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Initialize trainer
        trainer = FraudDetectionTrainer(models_dir=args.models_dir)
        
        # Configure models for quick training
        if args.quick_train:
            # Reduce model complexity for faster training
            trainer.model_configs['random_forest']['params'].update({
                'n_estimators': 50,
                'max_depth': 10
            })
            trainer.model_configs['gradient_boosting']['params'].update({
                'n_estimators': 50,
                'max_depth': 4
            })
            logger.info("Reduced model complexity for quick training")
        
        # Run training pipeline
        logger.info("Starting training pipeline...")
        results = trainer.train_pipeline(
            users_file=args.users_file,
            transactions_file=args.transactions_file,
            sample_size=sample_size
        )
        
        # Display results
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Model performance summary
        logger.info("Model Performance Summary:")
        logger.info("-" * 40)
        
        best_model = results['registry_info']['best_model']
        for model_name, metrics in results['registry_info']['metrics'].items():
            status = " (BEST)" if model_name == best_model else ""
            logger.info(f"{model_name}{status}:")
            logger.info(f"  AUC: {metrics['auc']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            
            if 'cv_auc_mean' in metrics:
                logger.info(f"  CV AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']*2:.4f})")
            logger.info("")
        
        # Data information
        data_info = results['data_info']
        logger.info("Dataset Information:")
        logger.info("-" * 20)
        logger.info(f"Users: {data_info['users_shape']}")
        logger.info(f"Transactions: {data_info['transactions_shape']}")
        logger.info(f"Features: {data_info['features_shape']}")
        logger.info(f"Fraud Rate: {data_info['fraud_rate']:.4f} ({data_info['fraud_rate']*100:.2f}%)")
        
        # Model files
        logger.info("Saved Model Files:")
        logger.info("-" * 18)
        registry_info = results['registry_info']
        logger.info(f"Feature Engineer: {registry_info['feature_engineer_path']}")
        logger.info(f"Scaler: {registry_info['scaler_path']}")
        for model_name, model_path in registry_info['models'].items():
            logger.info(f"{model_name}: {model_path}")
        
        logger.info(f"\nModel registry: {args.models_dir}/model_registry.pkl")
        logger.info(f"Training completed at: {datetime.now()}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Check the logs above for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ Training completed successfully!")
        print(f"Best model: {results['registry_info']['best_model']}")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
