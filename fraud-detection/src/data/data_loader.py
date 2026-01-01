"""
Data loading utilities for fraud detection system.
Optimized for fast loading of large datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate fraud detection datasets with optimized loading."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filename: str = "fraud_data.csv", chunksize: Optional[int] = None) -> pd.DataFrame:
        """Load processed dataset with optional chunking for large files."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        if chunksize:
            # Load in chunks for memory efficiency
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=chunksize):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            # Use optimized dtypes for faster loading
            df = pd.read_csv(filepath, low_memory=False)
        
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def load_data_sample(
        self, 
        filename: str, 
        nrows: int = 100000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Load a random sample of data for fast EDA."""
        filepath = Path("data/raw") / filename if not Path(filename).is_absolute() else Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading sample ({nrows:,} rows) from {filepath}")
        
        # Get total rows (fast)
        total_rows = sum(1 for _ in open(filepath)) - 1
        
        if nrows >= total_rows:
            df = pd.read_csv(filepath, low_memory=False)
        else:
            # Random sample
            skip_rows = sorted(
                np.random.RandomState(random_state).choice(
                    range(1, total_rows + 1),
                    total_rows - nrows,
                    replace=False
                )
            )
            df = pd.read_csv(filepath, skiprows=skip_rows, low_memory=False)
        
        logger.info(f"Loaded {len(df):,} rows")
        return df
    
    def load_raw_data(self, filename: str, data_dir: str = "data/raw") -> pd.DataFrame:
        """Load raw dataset from CSV file."""
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        
        logger.info(f"Loading raw data from {filepath}")
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def split_features_target(
        self, 
        df: pd.DataFrame, 
        target_col: str = "is_fraud"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataset into features and target."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def validate_data(self, df: pd.DataFrame, target_col: str = "is_fraud") -> bool:
        """Validate dataset structure and data quality."""
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        if df[target_col].isna().any():
            logger.warning("Target column contains missing values")
        
        if df[target_col].dtype not in [int, float, bool]:
            raise ValueError(f"Target column must be numeric or boolean")
        
        logger.info("Data validation passed")
        return True

