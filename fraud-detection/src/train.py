"""
Main training script for bank fraud detection system.
Optimized for fast and secure execution.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import FraudDataPreprocessor
from src.models.model_trainer import FraudModelTrainer
from src.models.model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Starting Fraud Detection Model Training Pipeline")
    logger.info("="*60)
    
    # Load processed data
    try:
        logger.info("Loading preprocessed data splits...")
        X_train = pd.read_csv("data/processed/X_train.csv", low_memory=False)
        X_val = pd.read_csv("data/processed/X_val.csv", low_memory=False)
        X_test = pd.read_csv("data/processed/X_test.csv", low_memory=False)
        y_train = pd.read_csv("data/processed/y_train.csv", low_memory=False)['is_fraud']
        y_val = pd.read_csv("data/processed/y_val.csv", low_memory=False)['is_fraud']
        y_test = pd.read_csv("data/processed/y_test.csv", low_memory=False)['is_fraud']
        logger.info("✓ Preprocessed data loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"Preprocessed data not found: {e}")
        logger.info("Running data processing pipeline...")
        from src.data.data_processor import process_fraud_data
        
        # Process data (use sample for faster processing)
        df = process_fraud_data(sample_size=500000, random_state=42)
        
        # Encode transaction type if it exists
        if 'type' in df.columns:
            le = LabelEncoder()
            df['type_encoded'] = le.fit_transform(df['type'])
            Path("models").mkdir(exist_ok=True)
            joblib.dump(le, 'models/label_encoder.pkl')
            logger.info("✓ Transaction type encoded")
        
        # Split data
        feature_cols = [col for col in df.columns if col not in ['is_fraud', 'type']]
        X = df[feature_cols]
        y = df['is_fraud']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Save splits
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        X_train.to_csv("data/processed/X_train.csv", index=False)
        X_val.to_csv("data/processed/X_val.csv", index=False)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False, header=['is_fraud'])
        y_val.to_csv("data/processed/y_val.csv", index=False, header=['is_fraud'])
        y_test.to_csv("data/processed/y_test.csv", index=False, header=['is_fraud'])
        logger.info("✓ Data splits saved")
    
    logger.info(f"\nDataset Info:")
    logger.info(f"  Train: {X_train.shape[0]:,} samples")
    logger.info(f"  Validation: {X_val.shape[0]:,} samples")
    logger.info(f"  Test: {X_test.shape[0]:,} samples")
    logger.info(f"  Features: {X_train.shape[1]}")
    
    # Preprocessing pipeline
    logger.info("\n" + "="*60)
    logger.info("Fitting Preprocessing Pipeline")
    logger.info("="*60)
    preprocessor = FraudDataPreprocessor(
        use_scaling=True,
        scaler_type="robust",
        handle_imbalance=True,
        balance_method="smote",
        random_state=42
    )
    
    # Fit and transform training data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    logger.info(f"✓ Training data processed: {X_train_processed.shape}")
    
    # Transform validation and test sets (no balancing)
    preprocessor.handle_imbalance = False  # Don't balance validation/test
    X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    logger.info(f"✓ Validation and test data processed")
    
    # Save preprocessor
    Path("models").mkdir(exist_ok=True)
    preprocessor.save("models/preprocessor.pkl")
    logger.info("✓ Preprocessor saved")
    
    # Model training
    logger.info("\n" + "="*60)
    logger.info("Training Models (This may take several minutes...)")
    logger.info("="*60)
    trainer = FraudModelTrainer(random_state=42, scoring="recall")
    
    # Train selected models (faster models for demonstration)
    models_to_train = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    
    training_results = trainer.train_all_models(
        X_train_processed,
        y_train_processed,
        models_to_train=models_to_train,
        cv=3,  # Reduced for faster training
        n_iter=10  # Reduced for faster training
    )
    
    # Select best model based on validation recall
    logger.info("\n" + "="*60)
    logger.info("Selecting Best Model")
    logger.info("="*60)
    best_model_name, best_model = trainer.select_best_model(
        X_val_processed,
        y_val_processed,
        metric="recall"
    )
    
    # Save best model
    trainer.save_model(
        best_model,
        f"models/{best_model_name}_model.pkl",
        metadata={
            "model_name": best_model_name,
            "training_recall": training_results[best_model_name]["best_score"],
            "cv_mean": training_results[best_model_name]["cv_mean"],
            "cv_std": training_results[best_model_name]["cv_std"]
        }
    )
    logger.info(f"✓ Best model ({best_model_name}) saved")
    
    # Evaluation
    logger.info("\n" + "="*60)
    logger.info("Evaluating Models")
    logger.info("="*60)
    evaluator = ModelEvaluator(output_dir="reports/figures")
    
    # Evaluate on all sets
    logger.info("Generating predictions...")
    y_train_pred = best_model.predict(X_train_processed)
    y_train_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
    
    y_val_pred = best_model.predict(X_val_processed)
    y_val_pred_proba = best_model.predict_proba(X_val_processed)[:, 1]
    
    y_test_pred = best_model.predict(X_test_processed)
    y_test_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
    
    # Generate comprehensive evaluation report
    results = evaluator.generate_evaluation_report(
        y_train=y_train_processed.values,
        y_train_pred=y_train_pred,
        y_train_pred_proba=y_train_pred_proba,
        y_val=y_val_processed.values,
        y_val_pred=y_val_pred,
        y_val_pred_proba=y_val_pred_proba,
        y_test=y_test_processed.values,
        y_test_pred=y_test_pred,
        y_test_pred_proba=y_test_pred_proba,
        model_name=best_model_name,
        save_dir="reports/figures"
    )
    
    # Save evaluation results
    import json
    Path("reports").mkdir(exist_ok=True)
    with open("reports/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("✓ Evaluation results saved")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"\nBest Model: {best_model_name.replace('_', ' ').title()}")
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Recall:    {results['test']['recall']:.4f}")
    logger.info(f"  Precision: {results['test']['precision']:.4f}")
    logger.info(f"  F1-Score:  {results['test']['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {results['test']['roc_auc']:.4f}")
    logger.info(f"  Accuracy:  {results['test']['accuracy']:.4f}")
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
