"""
Data preprocessing pipeline for fraud detection.
Optimized for performance and security.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataPreprocessor:
    """Preprocessing pipeline optimized for fraud detection."""
    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create features required by the trained model.
        This MUST be identical for training and prediction.
        """
        X = X.copy()
        
        # Time-based features
        X['step_hour'] = X['step'] % 24
        X['step_day'] = X['step'] // 24
        X['is_weekend'] = (X['step_day'] % 7 >= 5).astype(int)
        
        # Amount-based features (vectorized for speed)
        X['amount_log'] = np.log1p(X['amount'])
        X['amount_sqrt'] = np.sqrt(X['amount'])
        X['amount_squared'] = X['amount'] ** 2
        
        # Balance difference features (vectorized)
        if 'oldbalanceOrg' in X.columns and 'newbalanceOrig' in X.columns:
            X['balance_change_org'] = X['newbalanceOrig'] - X['oldbalanceOrg']
            X['orig_zero_balance'] = (X['oldbalanceOrg'] == 0).astype(int)
        
        if 'oldbalanceDest' in X.columns and 'newbalanceDest' in X.columns:
            X['balance_change_dest'] = X['newbalanceDest'] - X['oldbalanceDest']
            X['dest_zero_balance'] = (X['oldbalanceDest'] == 0).astype(int)
        
        # Amount to balance ratio (with safe division)
        if 'amount' in X.columns and 'oldbalanceOrg' in X.columns:
            X['amount_balance_ratio'] = np.divide(
                X['amount'], 
                X['oldbalanceOrg'] + 1,  # +1 to avoid division by zero
                out=np.zeros_like(X['amount']),
                where=(X['oldbalanceOrg'] + 1) != 0
            )
        
        # Encode transaction type if it exists
        if 'type' in X.columns:
            # Load the label encoder used during training
            try:
                import joblib
                le = joblib.load('models/label_encoder.pkl')
                X['type_encoded'] = le.transform(X['type'])
            except FileNotFoundError:
                # Fallback: create simple encoding
                type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'CASH_IN': 3, 'DEBIT': 4}
                X['type_encoded'] = X['type'].map(type_mapping).fillna(0)
        
        return X
    
    def __init__(
        self,
        use_scaling: bool = True,
        scaler_type: str = "robust",  # robust or standard
        handle_imbalance: bool = True,
        balance_method: str = "smote",  # smote, undersample, or none
        random_state: int = 42
    ):
        self.use_scaling = use_scaling
        self.scaler_type = scaler_type
        self.handle_imbalance = handle_imbalance
        self.balance_method = balance_method
        self.random_state = random_state
        
        # Initialize components
        self.scaler = None
        self.imputer = SimpleImputer(strategy='median')
        self.balancer = None
        self.feature_names = None
        self.is_fitted = False
    
    def _validate_input(self, X: pd.DataFrame):
        """Validate input data for security and correctness."""
        if X.empty:
            raise ValueError("Input data is empty")
        
        # Check for infinite values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if np.isinf(X[numeric_cols]).any().any():
                logger.warning("Input contains infinite values, replacing with NaN")
                X = X.replace([np.inf, -np.inf], np.nan)
        
        # Check for suspiciously large values (potential data corruption)
        for col in numeric_cols:
            if X[col].abs().max() > 1e15:
                logger.warning(f"Column {col} contains extremely large values")
        
        return X
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit preprocessing pipeline on training data."""
        logger.info("Fitting preprocessing pipeline...")
        X = self._validate_input(X)
        X = self._feature_engineering(X)
        self.feature_names = X.columns.tolist()

        
        # Handle missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=self.feature_names,
            index=X.index
        )
        
        # Fit scaler
        if self.use_scaling:
            if self.scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(X_imputed)
        
        # Fit balancer (only on training data)
        if self.handle_imbalance and y is not None:
            if self.balance_method == "smote":
                # Use smaller k_neighbors for faster processing
                self.balancer = SMOTE(
                    random_state=42, 
                    k_neighbors=3,
                  # Parallel processing
                )
                X_scaled = self.scaler.transform(X_imputed) if self.use_scaling else X_imputed
                try:
                    self.balancer.fit_resample(X_scaled, y)
                except Exception as e:
                    logger.warning(f"SMOTE fitting issue: {e}. Will fit during transform.")
            elif self.balance_method == "undersample":
                self.balancer = RandomUnderSampler(random_state=self.random_state)
                X_scaled = self.scaler.transform(X_imputed) if self.use_scaling else X_imputed
                try:
                    self.balancer.fit_resample(X_scaled, y)
                except Exception as e:
                    logger.warning(f"Undersampling fitting issue: {e}. Will fit during transform.")
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transformation")
        
        X = self._validate_input(X)
        X = self._feature_engineering(X)

        
        # Ensure same columns as training
        if list(X.columns) != self.feature_names:
            # Reorder or select columns
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            X = X[self.feature_names]
        
        # Handle missing values
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=self.feature_names,
            index=X.index
        )
        
        # Scale features
        if self.use_scaling:
            X_scaled = self.scaler.transform(X_imputed)
            X_processed = pd.DataFrame(
                X_scaled,
                columns=self.feature_names,
                index=X.index
            )
        else:
            X_processed = X_imputed
        
        # Balance dataset (only on training data)
        if self.handle_imbalance and y is not None and self.balancer is not None:
            try:
                X_resampled, y_resampled = self.balancer.fit_resample(X_processed, y)
                X_processed = pd.DataFrame(X_resampled, columns=self.feature_names)
                y_processed = pd.Series(y_resampled)
                logger.info(f"Data balanced: {X_processed.shape[0]:,} samples")
            except Exception as e:
                logger.warning(f"Balancing failed: {e}. Using unbalanced data.")
                y_processed = y
        else:
            y_processed = y
        
        return X_processed, y_processed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit and transform data in one step."""
        self.fit(X, y)
        return self.transform(X, y)
    
    def save(self, filepath: str):
        """Save preprocessor to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FraudDataPreprocessor':
        """Load preprocessor from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        return joblib.load(filepath)

