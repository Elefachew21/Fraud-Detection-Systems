"""
Comprehensive model evaluation with metrics, curves, and analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation for fraud detection."""
    
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            metrics["average_precision"] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics (RMSE, MAE, R-squared)."""
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred)
        }
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """Plot and analyze confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Fraud", "Fraud"],
            yticklabels=["Non-Fraud", "Fraud"]
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate (Recall)", fontsize=12)
        plt.title(f"ROC Curve - {model_name}", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AP = {ap_score:.4f})")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"Precision-Recall Curve - {model_name}", fontsize=16, fontweight="bold")
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.close()
    
    def compare_models(
        self,
        models_results: Dict[str, Dict[str, float]],
        metric: str = "recall",
        save_path: Optional[str] = None
    ):
        """Compare multiple models using a bar chart."""
        model_names = list(models_results.keys())
        metric_values = [results.get(metric, 0) for results in models_results.values()]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(model_names, metric_values, color=sns.color_palette("husl", len(model_names)))
        plt.xlabel(metric.upper(), fontsize=12)
        plt.title(f"Model Comparison - {metric.upper()}", fontsize=16, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="x")
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            plt.text(val, i, f" {val:.4f}", va="center", fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison saved to {save_path}")
        
        plt.close()
    
    def error_analysis(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """Analyze misclassifications for error insights."""
        errors = pd.DataFrame(X.copy())
        errors["true_label"] = y_true
        errors["predicted_label"] = y_pred
        errors["is_correct"] = (y_true == y_pred)
        errors["error_type"] = "correct"
        errors.loc[(y_true == 0) & (y_pred == 1), "error_type"] = "false_positive"
        errors.loc[(y_true == 1) & (y_pred == 0), "error_type"] = "false_negative"
        
        logger.info(f"\nError Analysis for {model_name}:")
        logger.info(f"False Positives: {(errors['error_type'] == 'false_positive').sum()}")
        logger.info(f"False Negatives: {(errors['error_type'] == 'false_negative').sum()}")
        
        return errors
    
    def generate_evaluation_report(
        self,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_train_pred_proba: np.ndarray,
        y_val: np.ndarray,
        y_val_pred: np.ndarray,
        y_val_pred_proba: np.ndarray,
        y_test: Optional[np.ndarray] = None,
        y_test_pred: Optional[np.ndarray] = None,
        y_test_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "Model",
        save_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive evaluation report with all metrics and plots."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Calculate metrics for each set
        for set_name, y_true, y_pred, y_proba in [
            ("train", y_train, y_train_pred, y_train_pred_proba),
            ("val", y_val, y_val_pred, y_val_pred_proba),
            ("test", y_test, y_test_pred, y_test_pred_proba)
        ]:
            if y_true is None:
                continue
            
            metrics = self.calculate_classification_metrics(y_true, y_pred, y_proba)
            results[set_name] = metrics
            
            if save_dir:
                # Save plots
                self.plot_confusion_matrix(
                    y_true, y_pred,
                    model_name=f"{model_name} - {set_name}",
                    save_path=str(save_dir / f"confusion_matrix_{set_name}.png")
                )
                
                self.plot_roc_curve(
                    y_true, y_proba,
                    model_name=f"{model_name} - {set_name}",
                    save_path=str(save_dir / f"roc_curve_{set_name}.png")
                )
                
                self.plot_precision_recall_curve(
                    y_true, y_proba,
                    model_name=f"{model_name} - {set_name}",
                    save_path=str(save_dir / f"precision_recall_curve_{set_name}.png")
                )
        
        logger.info("\n" + "="*60)
        logger.info(f"EVALUATION REPORT - {model_name}")
        logger.info("="*60)
        
        for set_name, metrics in results.items():
            logger.info(f"\n{set_name.upper()} Set Metrics:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return results

