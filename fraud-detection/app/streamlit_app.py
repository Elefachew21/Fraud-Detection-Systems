"""
Streamlit Dashboard for Bank Fraud Detection System
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_preprocessor import FraudDataPreprocessor

# Page config
st.set_page_config(
    page_title="Bank Fraud Detection System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor."""
    try:
        # Load the model (any one of your trained models)
        model_paths = [
            "models/xgboost_model.pkl",
            "models/lightgbm_model.pkl",
            "models/random_forest_model.pkl",
            "models/logistic_regression_model.pkl"
        ]
        model = None
        model_name = None
        for path in model_paths:
            if Path(path).exists():
                model = joblib.load(path)
                model_name = Path(path).stem.replace("_model", "")
                break
        
        if model is None:
            st.error("No trained model found. Please train a model first.")
            return None, None, None
        
        # Load preprocessor
        preprocessor_path = "models/preprocessor.pkl"
        if not Path(preprocessor_path).exists():
            st.error("Preprocessor not found. Please create and save it first.")
            return None, None, None
        
        preprocessor = FraudDataPreprocessor.load(preprocessor_path)
        return model, preprocessor, model_name
    
    except Exception as e:
        st.error(f"Error loading model/preprocessor: {str(e)}")
        return None, None, None


def main():
    """Main Streamlit app."""
    st.markdown('<p class="main-header">🏦 Bank Fraud Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Home",
        "Interactive Prediction",
        "EDA Dashboard",
        "Model Performance"
    ])
    
    if page == "Home":
        show_home()
    elif page == "Interactive Prediction":
        show_prediction()
    elif page == "EDA Dashboard":
        show_eda()
    elif page == "Model Performance":
        show_model_performance()


def show_home():
    """Home page."""
    st.header("Welcome to the Bank Fraud Detection System")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", "6.3M+")
    with col2:
        st.metric("Detection Accuracy", "99.9%+")
    with col3:
        st.metric("Fraud Recall", "Optimized")
    
    st.markdown("---")
    st.subheader("Features")
    st.markdown("""
    - **Real-time Fraud Detection**: Predict fraud in real-time
    - **Multiple ML Models**: XGBoost, LightGBM, Random Forest, Logistic Regression
    - **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrix
    - **Feature Importance Analysis**: Understand model decisions
    - **Interactive Dashboard**: Explore data and model insights
    """)


def show_prediction():
    """Interactive prediction page."""
    st.header("Interactive Fraud Prediction")
    
    model, preprocessor, model_name = load_model_and_preprocessor()
    if model is None:
        return
    
    st.info(f"Using model: **{model_name.replace('_', ' ').title()}**")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Step (Time)", min_value=1, max_value=744, value=1)
            transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
            amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
            oldbalance_org = st.number_input("Old Balance Origin", min_value=0.0, value=1000.0, format="%.2f")
            newbalance_org = st.number_input("New Balance Origin", min_value=0.0, value=0.0, format="%.2f")
        
        with col2:
            oldbalance_dest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0, format="%.2f")
            newbalance_dest = st.number_input("New Balance Destination", min_value=0.0, value=1000.0, format="%.2f")
        
        submitted = st.form_submit_button("Predict Fraud", type="primary")
    
    if submitted:
        try:
            raw_input = pd.DataFrame([{
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalance_org,
                "newbalanceOrig": newbalance_org,
                "oldbalanceDest": oldbalance_dest,
                "newbalanceDest": newbalance_dest
            }])
            
            # Transform features (all engineered features handled by preprocessor)
            feature_processed, _ = preprocessor.transform(raw_input)
            
            # Predict
            prediction = model.predict(feature_processed)[0]
            probability = model.predict_proba(feature_processed)[0]
            
            # Display
            col1, col2, col3 = st.columns(3)
            with col1:
                if prediction == 1:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
            with col2:
                st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
            with col3:
                st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")
            
            st.progress(probability[1])
            st.caption(f"Fraud Risk Score: {probability[1]:.4f}")
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


def show_eda():
    """EDA dashboard."""
    st.header("📊 Exploratory Data Analysis")
    
    try:
        df_sample = pd.read_csv("data/raw/Fraud.csv", nrows=50000)
        
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{len(df_sample):,}")
        with col2:
            st.metric("Fraud Cases", f"{df_sample['isFraud'].sum():,}")
        with col3:
            st.metric("Fraud Rate", f"{(df_sample['isFraud'].sum()/len(df_sample)*100):.2f}%")
        with col4:
            st.metric("Features", len(df_sample.columns))
        
        # Class distribution
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sample['isFraud'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class (0=Non-Fraud, 1=Fraud)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        # Transaction type analysis
        st.subheader("Transaction Type Analysis")
        fraud_by_type = df_sample.groupby('type')['isFraud'].agg(['count', 'sum', 'mean'])
        fraud_by_type.columns = ['Total', 'Fraud_Count', 'Fraud_Rate']
        st.dataframe(fraud_by_type.sort_values('Fraud_Rate', ascending=False))
        
        # Amount distribution
        st.subheader("Amount Distribution")
        fraud_amounts = df_sample[df_sample['isFraud'] == 1]['amount']
        normal_amounts = df_sample[df_sample['isFraud'] == 0]['amount']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist([normal_amounts, fraud_amounts], bins=50, label=['Normal', 'Fraud'], alpha=0.7, color=['skyblue', 'salmon'])
        ax.set_xlabel('Transaction Amount')
        ax.set_ylabel('Frequency')
        ax.set_title('Amount Distribution by Fraud Status')
        ax.legend()
        ax.set_yscale('log')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


def show_model_performance():
    """Model performance page."""
    st.header("📈 Model Performance")
    
    try:
        import json
        with open("reports/evaluation_results.json", "r") as f:
            results = json.load(f)
        
        st.subheader("Performance Metrics")
        sets = ["train", "val", "test"]
        metrics_to_show = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        
        for set_name in sets:
            if set_name in results:
                st.write(f"### {set_name.upper()} Set")
                cols = st.columns(len(metrics_to_show))
                for i, metric in enumerate(metrics_to_show):
                    if metric in results[set_name]:
                        cols[i].metric(metric.replace("_", " ").title(),
                                       f"{results[set_name][metric]:.4f}")
        
        st.subheader("Confusion Matrices")
        for set_name in sets:
            img_path = f"reports/figures/confusion_matrix_{set_name}.png"
            if Path(img_path).exists():
                st.write(f"### {set_name.upper()} Set")
                st.image(img_path)
        
    except FileNotFoundError:
        st.warning("Evaluation results not found. Please train and evaluate models first.")
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")


if __name__ == "__main__":
    main()
