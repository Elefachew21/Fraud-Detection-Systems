"""
Visualization utilities for fraud detection analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from pathlib import Path

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


class PlotUtils:
    """Utility class for creating visualizations."""
    
    @staticmethod
    def plot_class_distribution(y: pd.Series, title: str = "Class Distribution", save_path: Optional[str] = None):
        """Plot class distribution."""
        plt.figure(figsize=(10, 6))
        y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close()
    
    @staticmethod
    def plot_feature_importance(importance: pd.Series, top_n: int = 20, save_path: Optional[str] = None):
        """Plot feature importance."""
        top_features = importance.head(top_n).sort_values(ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel("Importance", fontsize=12)
        plt.title(f"Top {top_n} Feature Importance", fontsize=16, fontweight="bold")
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot correlation matrix heatmap."""
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
        plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close()
    
    @staticmethod
    def plot_amount_distribution(df: pd.DataFrame, fraud_col: str = "isFraud", save_path: Optional[str] = None):
        """Plot transaction amount distribution by fraud status."""
        plt.figure(figsize=(12, 6))
        
        fraud = df[df[fraud_col] == 1]['amount']
        normal = df[df[fraud_col] == 0]['amount']
        
        plt.hist([normal, fraud], bins=50, label=['Normal', 'Fraud'], alpha=0.7, color=['skyblue', 'salmon'])
        plt.xlabel("Transaction Amount", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Transaction Amount Distribution by Fraud Status", fontsize=16, fontweight="bold")
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close()

