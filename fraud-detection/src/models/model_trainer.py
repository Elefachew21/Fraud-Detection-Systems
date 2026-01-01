"""
Model training utilities with multiple algorithms.
Optimized for fraud detection with recall focus and fast training.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, Any, List
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """Train and optimize multiple models for fraud detection."""
    
    def __init__(self, random_state: int = 42, scoring: str = "recall"):
        self.random_state = random_state
        self.scoring = scoring  # Focus on recall for fraud detection
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_history = {}
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for different models (optimized for speed)."""
        return {
            "logistic_regression": {
                "model": LogisticRegression(
                    random_state=self.random_state, 
                    max_iter=1000,
                    n_jobs=-1,  # Parallel processing
                    solver='lbfgs'  # Fast solver
                ),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "class_weight": [None, "balanced"]
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    random_state=self.random_state, 
                    n_jobs=-1,
                    verbose=0
                ),
                "params": {
                    "n_estimators": [100, 200],  # Reduced for faster training
                    "max_depth": [15, 20, None],
                    "min_samples_split": [2, 5],
                    "class_weight": [None, "balanced"]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(
                    random_state=self.random_state,
                    verbose=0
                ),
                "params": {
                    "n_estimators": [100, 150],  # Reduced for speed
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                }
            },
            "xgboost": {
                "model": XGBClassifier(
                    random_state=self.random_state, 
                    eval_metric='logloss', 
                    n_jobs=-1,
                    tree_method='hist',  # Faster training
                    verbosity=0
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.05, 0.1],
                    "scale_pos_weight": [1, 3, 5]  # Handle imbalance
                }
            },
            "lightgbm": {
                "model": XGBClassifier(
                    random_state=self.random_state, 
                    verbose=-1, 
                    n_jobs=-1,
                    boosting_type='gbdt'
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.05, 0.1],
                    "class_weight": [None, "balanced"]
                }
            },
            "svm": {
                "model": SVC(
                    random_state=self.random_state, 
                    probability=True,
                    cache_size=500  # For faster training
                ),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "class_weight": [None, "balanced"]
                }
            }
        }
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 3,  # Reduced CV for speed
        n_iter: int = 20,
        use_random_search: bool = True
    ) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning."""
        logger.info(f"Training {model_name}...")
        
        configs = self.get_model_configs()
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = configs[model_name]
        base_model = config["model"]
        param_grid = config["params"]
        
        # Use RandomizedSearchCV for faster training
        if use_random_search and n_iter:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=self.scoring,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        else:
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        
        model = search.best_estimator_
        self.models[model_name] = model
        
        # Get cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=-1)
        
        result = {
            "model": model,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
        
        logger.info(f"{model_name} trained. Best CV {self.scoring}: {search.best_score_:.4f}")
        return result
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models_to_train: Optional[List[str]] = None,
        cv: int = 3,
        n_iter: int = 20
    ) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and compare performance."""
        configs = self.get_model_configs()
        
        if models_to_train is None:
            models_to_train = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
        
        results = {}
        for model_name in models_to_train:
            try:
                result = self.train_model(model_name, X_train, y_train, cv, n_iter)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.training_history = results
        return results
    
    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model and return comprehensive metrics."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
        
        logger.info(f"{set_name} set - Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        
        return metrics
    
    def select_best_model(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = "recall"
    ) -> Tuple[str, Any]:
        """Select best model based on validation performance."""
        logger.info(f"Selecting best model based on {metric}...")
        
        best_score = -np.inf
        best_name = None
        best_model = None
        
        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_val, y_val, f"{name}_val")
            score = metrics.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
        
        self.best_model = best_model
        self.best_model_name = best_name
        
        logger.info(f"Best model: {best_name} with {metric}={best_score:.4f}")
        return best_name, best_model
    
    def save_model(self, model: Any, filepath: str, metadata: Optional[Dict] = None):
        """Save model and metadata."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath, compress=3)  # Compression for smaller files
        logger.info(f"Model saved to {filepath}")
        
        if metadata:
            metadata_path = str(filepath).replace(".pkl", "_metadata.json")
            metadata["saved_at"] = datetime.now().isoformat()
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)

