# 🏦 Bank Fraud Detection System

A comprehensive, production-ready machine learning system for detecting fraudulent bank transactions. This system is optimized for **speed**, **security**, and **competitive performance** in real-world applications.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Security Considerations](#security-considerations)
- [Performance Optimizations](#performance-optimizations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This fraud detection system analyzes bank transaction data to identify fraudulent activities with high accuracy and recall. The system uses multiple machine learning algorithms and is optimized for handling large-scale datasets efficiently.

### Key Objectives

- **High Recall**: Optimize for detecting maximum fraud cases (minimize false negatives)
- **Fast Processing**: Efficient data loading and model inference
- **Secure**: Input validation and safe data handling
- **Production-Ready**: Comprehensive evaluation and monitoring

## ✨ Features

- 🔍 **Multiple ML Models**: XGBoost, LightGBM, Random Forest, Logistic Regression
- 📊 **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrix, F1-Score
- 🎯 **Recall Optimization**: Focus on detecting fraud cases (critical for financial security)
- ⚡ **Performance Optimized**: Fast data loading, chunking, and parallel processing
- 🔒 **Security Features**: Input validation, data sanitization, secure model loading
- 📈 **Interactive Dashboard**: Streamlit web application for real-time predictions
- 📓 **Detailed Notebooks**: Complete EDA, preprocessing, modeling, and evaluation workflows
- 📉 **Visualization**: Rich plots and analysis reports

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended for large datasets

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd fraud-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
```

## 🏃 Quick Start

### 1. Data Processing

Place your raw data file (`fraud.csv`) in `data/raw/` directory.

Process the data (with sampling for faster processing):

```bash
python -m src.data.data_processor
```

Or process full dataset:

```python
from src.data.data_processor import process_fraud_data
df = process_fraud_data(sample_size=None)  # Use full dataset
```

### 2. Train Models

Train all models with optimized hyperparameters:

```bash
python src/train.py
```

This will:
- Load and preprocess data
- Train multiple models (XGBoost, LightGBM, Random Forest, Logistic Regression)
- Select the best model based on validation recall
- Generate comprehensive evaluation reports
- Save models and preprocessing pipeline

### 3. Run Interactive Dashboard

Launch the Streamlit dashboard:

```bash
python -m streamlit run app/streamlit_app.py
```

Access the dashboard at `http://localhost:8501`

## 📁 Project Structure

```
fraud-detection/
├── data/                          # Data directory
│   ├── raw/                       # Original, immutable data
│   │   └── fraud.csv              # Raw transaction data
│   ├── processed/                 # Cleaned, transformed data
│   │   ├── fraud_data.csv         # Processed dataset
│   │   ├── X_train.csv            # Training features
│   │   ├── X_val.csv              # Validation features
│   │   ├── X_test.csv             # Test features
│   │   ├── y_train.csv            # Training labels
│   │   ├── y_val.csv              # Validation labels
│   │   └── y_test.csv             # Test labels
│   └── external/                  # Third-party data sources
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Data cleaning & preprocessing
│   ├── 03_modeling.ipynb         # Model development
│   └── 04_evaluation.ipynb       # Model evaluation & analysis
│
├── src/                           # Source code
│   ├── data/                      # Data processing modules
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── data_preprocessor.py  # Preprocessing pipeline
│   │   └── data_processor.py     # Main data processing script
│   ├── features/                  # Feature engineering
│   │   └── feature_engineering.py # Feature creation utilities
│   ├── models/                    # Model building & evaluation
│   │   ├── model_trainer.py      # Model training utilities
│   │   └── model_evaluator.py    # Evaluation & metrics
│   ├── visualization/             # Plotting utilities
│   │   └── plot_utils.py         # Visualization helpers
│   └── train.py                   # Main training script
│
├── models/                        # Saved models
│   ├── *_model.pkl               # Serialized trained models
│   ├── preprocessor.pkl          # Preprocessing pipeline
│   ├── label_encoder.pkl         # Label encoders
│   └── *_metadata.json           # Model metadata
│
├── reports/                       # Generated reports
│   ├── figures/                  # Saved visualizations
│   │   ├── confusion_matrix_*.png
│   │   ├── roc_curve_*.png
│   │   └── precision_recall_curve_*.png
│   └── evaluation_results.json   # Evaluation metrics
│
├── app/                           # Web application
│   └── streamlit_app.py          # Streamlit dashboard
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 💻 Usage

### Data Processing

```python
from src.data.data_processor import process_fraud_data

# Process with sampling (faster for large datasets)
df = process_fraud_data(
    input_path="data/raw/fraud.csv",
    output_path="data/processed/fraud_data.csv",
    sample_size=500000,  # Use None for full dataset
    random_state=42
)
```

### Model Training

```python
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import FraudDataPreprocessor
from src.models.model_trainer import FraudModelTrainer

# Load data
loader = DataLoader()
X_train = loader.load_data("X_train.csv")
y_train = loader.load_data("y_train.csv")['is_fraud']

# Preprocess
preprocessor = FraudDataPreprocessor(
    handle_imbalance=True,
    balance_method="smote"
)
X_train_proc, y_train_proc = preprocessor.fit_transform(X_train, y_train)

# Train models
trainer = FraudModelTrainer(scoring="recall")
results = trainer.train_all_models(X_train_proc, y_train_proc)
best_model_name, best_model = trainer.select_best_model(X_val, y_val)
```

### Model Evaluation

```python
from src.models.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.generate_evaluation_report(
    y_train, y_train_pred, y_train_pred_proba,
    y_val, y_val_pred, y_val_pred_proba,
    y_test, y_test_pred, y_test_pred_proba,
    model_name="XGBoost"
)
```

### Making Predictions

```python
import joblib

# Load model and preprocessor
model = joblib.load("models/xgboost_model.pkl")
preprocessor = FraudDataPreprocessor.load("models/preprocessor.pkl")

# Preprocess new data
X_new_proc, _ = preprocessor.transform(X_new)

# Predict
predictions = model.predict(X_new_proc)
probabilities = model.predict_proba(X_new_proc)
```

## 📊 Model Performance

The system evaluates models using multiple metrics:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives) ⭐ **Optimized**
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Average Precision**: Area under the Precision-Recall curve

### Visualization Outputs
- Confusion Matrices
- ROC Curves
- Precision-Recall Curves
- Feature Importance Plots
- Model Comparison Charts

### Expected Performance

With proper hyperparameter tuning and class balancing:
- **Recall**: > 0.85 (optimized for fraud detection)
- **ROC-AUC**: > 0.95
- **F1-Score**: > 0.80
- **Precision**: Varies based on recall optimization trade-off

## 🔍 Key Findings

### Data Characteristics
- **Highly Imbalanced Dataset**: Fraud cases represent < 1% of transactions
- **Transaction Types**: TRANSFER and CASH_OUT show higher fraud rates
- **Balance Patterns**: Zero balances and large balance changes are fraud indicators
- **Amount Patterns**: Fraud transactions often have distinct amount distributions

### Model Insights
- **XGBoost/LightGBM**: Best performance for fraud detection
- **SMOTE Balancing**: Effective for handling class imbalance
- **Feature Engineering**: Balance changes and ratios are highly predictive
- **Recall Optimization**: Critical for minimizing financial losses

## 🔒 Security Considerations

### Input Validation
- All inputs are validated before processing
- Infinite and extreme values are handled safely
- Data type checking and sanitization

### Model Security
- Secure model loading with file existence checks
- Metadata validation
- Safe prediction interfaces

### Data Privacy
- No sensitive data logged
- Secure data handling in preprocessing
- Option to use encrypted storage for models

## ⚡ Performance Optimizations

### Fast Loading
- **Chunking**: Process large files in chunks
- **Sampling**: Fast EDA with random sampling
- **Optimized Dtypes**: Memory-efficient data types
- **Parallel Processing**: Multi-threading for model training

### Efficient Training
- **RandomizedSearchCV**: Faster hyperparameter tuning
- **Reduced CV Folds**: 3-fold CV for speed
- **Early Stopping**: For gradient boosting models
- **Histogram-based Trees**: XGBoost/LightGBM optimizations

### Model Inference
- **Compressed Models**: Joblib compression for smaller files
- **Batch Prediction**: Efficient batch processing
- **Caching**: Streamlit caching for faster dashboard

## 📚 Notebooks Guide

### 01_eda.ipynb
- Dataset exploration
- Class distribution analysis
- Feature distributions
- Correlation analysis
- Visualizations

### 02_preprocessing.ipynb
- Data cleaning steps
- Feature engineering
- Handling missing values
- Train/validation/test split

### 03_modeling.ipynb
- Model training workflows
- Hyperparameter tuning
- Model comparison
- Best model selection

### 04_evaluation.ipynb
- Comprehensive evaluation metrics
- Confusion matrix analysis
- ROC and Precision-Recall curves
- Error analysis
- Feature importance

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Bank transaction fraud detection dataset
- Libraries: scikit-learn, XGBoost, LightGBM, Streamlit
- Community: Open source ML community

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This system is optimized for fraud detection with a focus on recall. Adjust the scoring metric in `FraudModelTrainer` if you need to balance precision and recall differently.

