"""
Data processing script for bank fraud detection.
Optimized for handling large datasets efficiently.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional
from src.features.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_fraud_data(
    input_path: str = "data/raw/fraud.csv",
    output_path: str = "data/processed/fraud_data.csv",
    sample_size: Optional[int] = None,
    random_state: int = 42,
    use_chunks: bool = False,
    chunk_size: int = 100000
) -> pd.DataFrame:
    """
    Process raw fraud detection data efficiently.
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed data
        sample_size: If provided, sample this many rows
        random_state: Random seed for reproducibility
        use_chunks: Process in chunks for very large files
        chunk_size: Size of chunks if use_chunks is True
    """
    logger.info(f"Loading data from {input_path}...")
    
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # For large datasets, use chunking or sampling
    if sample_size:
        logger.info(f"Sampling {sample_size:,} rows from dataset...")
        # Fast row counting
        total_rows = sum(1 for _ in open(input_file)) - 1
        
        if sample_size >= total_rows:
            df = pd.read_csv(input_file, low_memory=False)
        else:
            # Random sample using skiprows (memory efficient)
            skip_rows = sorted(
                np.random.RandomState(random_state).choice(
                    range(1, total_rows + 1),
                    total_rows - sample_size,
                    replace=False
                )
            )
            df = pd.read_csv(input_file, skiprows=skip_rows, low_memory=False)
        logger.info(f"Loaded {len(df):,} sampled rows")
    
    elif use_chunks:
        logger.info(f"Processing in chunks of {chunk_size:,} rows...")
        chunks = []
        for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {(i + 1) * chunk_size:,} rows...")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df):,} rows in chunks")
    
    else:
        # Try to load full dataset
        try:
            df = pd.read_csv(input_file, low_memory=False)
            logger.info(f"Loaded {len(df):,} rows")
        except MemoryError:
            logger.warning("Memory error. Using sample of 500k rows...")
            sample_size = 500000
            total_rows = sum(1 for _ in open(input_file)) - 1
            skip_rows = sorted(
                np.random.RandomState(random_state).choice(
                    range(1, total_rows + 1),
                    total_rows - sample_size,
                    replace=False
                )
            )
            df = pd.read_csv(input_file, skiprows=skip_rows, low_memory=False)
            logger.info(f"Loaded {len(df):,} sampled rows")
    
    logger.info("Starting data preprocessing...")
    
    # Rename target column for consistency
    if 'isFraud' in df.columns:
        df.rename(columns={'isFraud': 'is_fraud'}, inplace=True)
    
    # Create feature engineer instance
    fe = FeatureEngineer()
    
    # Create features efficiently
    df = fe.create_transaction_features(df)
    df = fe.create_interaction_features(df)
    
    # Encode transaction type if it exists
    if 'type' in df.columns:
        df['type_encoded'] = df['type'].astype('category').cat.codes
    
    # Drop unnecessary columns (IDs that don't provide useful information)
    columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Optimize data types for memory efficiency
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    logger.info(f"Processed data shape: {df.shape}")
    fraud_count = df['is_fraud'].sum()
    logger.info(f"Fraud cases: {fraud_count:,} ({(fraud_count/len(df)*100):.4f}%)")
    
    # Save processed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Process data with sample for faster processing
    df = process_fraud_data(sample_size=500000)
    print(f"\nProcessed dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
