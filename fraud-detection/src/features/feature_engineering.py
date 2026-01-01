"""
Feature engineering utilities for fraud detection.
Optimized for performance.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
"""Feature engineering for fraud detection with performance optimizations."""

def __init__(self):
    self.feature_stats = {}

def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction-based features efficiently."""
    df = df.copy()
    
    # Time-based features (optimized)
    if 'step' in df.columns:
        df['step_hour'] = df['step'] % 24
        df['step_day'] = df['step'] // 24
        df['is_weekend'] = (df['step_day'] % 7 >= 5).astype(int)
    
    # Amount-based features (vectorized for speed)
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_sqrt'] = np.sqrt(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
    
    # Balance difference features (vectorized)
    if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
        df['balance_change_org'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['orig_zero_balance'] = (df['oldbalanceOrg'] == 0).astype(int)
    
    if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['dest_zero_balance'] = (df['oldbalanceDest'] == 0).astype(int)
    
    # Amount to balance ratio (with safe division)
    if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
        df['amount_balance_ratio'] = np.divide(
            df['amount'], 
            df['oldbalanceOrg'] + 1,  # +1 to avoid division by zero
            out=np.zeros_like(df['amount']),
            where=(df['oldbalanceOrg'] + 1) != 0
        )
    
    return df

def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features efficiently."""
    df = df.copy()
    
    # Type-amount interaction (if type exists)
    if 'type_encoded' in df.columns and 'amount' in df.columns:
        df['amount_type_interaction'] = df['amount'] * df['type_encoded']
    
    return df

def create_statistical_features(
    self, 
    df: pd.DataFrame, 
    group_cols: List[str], 
    value_col: str
) -> pd.DataFrame:
    """Create statistical features grouped by columns (optimized with groupby)."""
    df = df.copy()
    
    for group_col in group_cols:
        if group_col in df.columns:
            grouped = df.groupby(group_col)[value_col].agg(['mean', 'std', 'max', 'min'])
            
            df[f'{group_col}_{value_col}_mean'] = df[group_col].map(grouped['mean'])
            df[f'{group_col}_{value_col}_std'] = df[group_col].map(grouped['std'])
            df[f'{group_col}_{value_col}_max'] = df[group_col].map(grouped['max'])
            df[f'{group_col}_{value_col}_min'] = df[group_col].map(grouped['min'])
    
    return df

def select_features(
    self,
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    method: str = "all",
    top_k: Optional[int] = None
) -> List[str]:
    """Select features based on correlation or importance."""
    if method == "all":
        feature_cols = [col for col in df.columns if col != target_col]
    elif method == "correlation":
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        feature_cols = correlations.drop(target_col).head(top_k or len(correlations)).index.tolist()
    else:
        feature_cols = [col for col in df.columns if col != target_col]
    
    return feature_cols

def remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """Remove highly correlated features efficiently."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return to_drop

