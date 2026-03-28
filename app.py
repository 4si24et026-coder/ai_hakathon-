"""
High-Value Customer Predictor - Streamlit App
==============================================
INSTRUCTIONS TO RUN LOCALLY:
1. Open your terminal.
2. Install the necessary Python packages:
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
3. Start the Streamlit server:
   streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Page Configuration
st.set_page_config(page_title="High-Value Customer Predictor", layout="wide", page_icon="📈")

# ==========================================
# 1. Data Processing & Caching
# ==========================================
@st.cache_data
def load_and_preprocess_data(filepath="hackathon_dataset.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Dataset '{filepath}' not found!")
        st.stop()
        
    if 'high_value_purchase' not in df.columns:
        st.error("Target variable 'high_value_purchase' not found in dataset!")
        st.stop()

    # Handle Missing Values: Impute medians/modes and add missingness flags
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    for col in cols_with_missing:
        df[f"{col}_is_missing"] = df[col].isnull().astype(int)
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    # Fix Invalid Entries
    if 'age' in df.columns:
        df['age'] = df['age'].clip(lower=18, upper=100) # Ages < 18 or > 100
    if 'avg_review_rating' in df.columns:
        df['avg_review_rating'] = df['avg_review_rating'].clip(lower=1.0, upper=5.0) # Ratings outside 1-5
        
    # Feature Engineering
    if 'total_purchases' in df.columns and 'avg_order_value' in df.columns:
        df['estimated_clv'] = df['total_purchases'] * df['avg_order_value'] # Proxy for Customer Lifetime Value
        
    if 'account_age_months' in df.columns and 'total_purchases' in df.columns:
        df['purchase_frequency'] = df['total_purchases'] / (df['account_age_months'].replace(0, 1)) # Frequency
        
    if 'email_opens' in df.columns and 'bounce_rate' in df.columns:
        df['engagement_score'] = df['email_opens'] * (1 - df['bounce_rate']) # Engagement Score
        
    return df

@st.cache_data
def prepare_features(df):
    target = 'high_value_purchase'
    # Drop identifier columns uniquely
    cols_to_drop = ['customer_id'] if 'customer_id' in df.columns else []
    
    X = df.drop(columns=[target] + cols_to_drop)
    y = df[target]
    
    # Extract categorical columns for encoding
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-Hot Encoding for categorical features implicitly handles them
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    return X_encoded, y, cat_cols, cols_to_drop

# ==========================================
# 2. Model Training & Evaluation (Background)
# ==========================================
@st.cache_resource
def train_model(X, y):
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model Training: Random Forest Classifier (Optimized via class weighting)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Test Set Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    metrics = {
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    return model, metrics, X_train.columns.tolist()

# ==========================================
# Application Setup
# ==========================================
# Load and setup data/model globally
df_raw = load_and_preprocess_data()
X_encoded, y, cat_cols_list, dropped_cols = prepare_features(df_raw)
model, test_metrics, model_features = train_model(X_encoded, y)

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Project Insights", "Make a Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("This application predicts if a customer is likely to make a high-value purchase based on historical behavior and engineered features.")

# ==========================================
# Section 1: Insights (EDA & Model Metrics)
# ==========================================
if menu == "Project Insights":
    st.title("📊 Project Insights & Model Dashboard")
    st.markdown("Explore the exploratory data analysis and final machine learning model performance.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='high_value_purchase', data=df_raw, palette='viridis', ax=ax)
        ax.set_xticklabels(['Unlikely (0)', 'Likely (1)'])
        ax.set_title("High-Value Purchase Classes")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Key Feature: Estimated CLV Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df_raw, x='estimated_clv', hue='high_value_purchase', kde=True, palette='viridis', ax=ax2, element="step")
        ax2.set_title("Customer Lifetime Value vs Target")
        st.pyplot(fig2)
        
    st.markdown("---")
    st.subheader("🚀 Final Model Performance (Test Set)")
    st.markdown("The Random Forest Classifier was trained to optimize for recognizing both classes effectively.")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("F1-Score", f"{test_metrics['F1-Score']:.4f}")
    m_col2.metric("Precision", f"{test_metrics['Precision']:.4f}")
    m_col3.metric("Recall", f"{test_metrics['Recall']:.4f}")
    m_col4.metric("ROC-AUC", f"{test_metrics['ROC-AUC']:.4f}")

    if hasattr(model, 'feature_importances_'):
        st.subheader("Top Predictors")
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:] # Top 10
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.barh(range(len(indices)), importances[indices], align='center')
        ax3.set_yticks(range(len(indices)), [model_features[i] for i in indices])
        ax3.set_title("Top 10 Feature Importances")
        st.pyplot(fig3)

# ==========================================
# Section 2: Interactive Predictor
# ==========================================
elif menu == "Make a Prediction":
    st.title("🔮 Interactive Customer Predictor")
    st.markdown("Enter hypothetical customer details below to predict their likelihood of making a high-value purchase.")
    
    # We dynamically create form fields based on the raw dataset's unencoded features (excluding target & engineered)
    base_features = list(df_raw.drop(columns=['high_value_purchase', 'estimated_clv', 'purchase_frequency', 'engagement_score'] + dropped_cols + [col for col in df_raw.columns if col.endswith('_is_missing')]).columns)
    
    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)
        input_data = {}
        
        for i, col in enumerate(base_features):
            current_col = [col_a, col_b, col_c][i % 3]
            with current_col:
                if df_raw[col].dtype in ['int64', 'float64']:
                    min_val = float(df_raw[col].min())
                    max_val = float(df_raw[col].max())
                    mean_val = float(df_raw[col].mean())
                    # Ensure integer steps for int columns
                    step = 1.0 if df_raw[col].dtype == 'int64' else float((max_val - min_val) / 100)
                    input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, step=step)
                else:
                    unique_vals = df_raw[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", unique_vals)
                    
        submit_button = st.form_submit_button(label="Predict High-Value Purchase")
        
    if submit_button:
        # Convert user input to DataFrame
        user_df = pd.DataFrame([input_data])
        
        # Apply the SAME feature engineering steps manually
        if 'total_purchases' in user_df.columns and 'avg_order_value' in user_df.columns:
            user_df['estimated_clv'] = user_df['total_purchases'] * user_df['avg_order_value']
            
        if 'account_age_months' in user_df.columns and 'total_purchases' in user_df.columns:
            user_df['purchase_frequency'] = user_df['total_purchases'] / max(user_df['account_age_months'].iloc[0], 1)
            
        if 'email_opens' in user_df.columns and 'bounce_rate' in user_df.columns:
            user_df['engagement_score'] = user_df['email_opens'] * (1 - user_df['bounce_rate'])
            
        # Add missingness flags manually (always 0 since user enforces valid inputs in the UI)
        missingness_cols = [col for col in df_raw.columns if col.endswith('_is_missing')]
        for missing_col in missingness_cols:
            user_df[missing_col] = 0
            
        # One-Hot Encode user inputs to match the trained model's features structure
        user_encoded = pd.get_dummies(user_df, columns=cat_cols_list)
        
        # Align features: Add missing columns with 0, drop extra columns
        for feature in model_features:
            if feature not in user_encoded.columns:
                user_encoded[feature] = 0
        user_encoded = user_encoded[model_features]
        
        # Make Prediction
        prediction = model.predict(user_encoded)[0]
        probability = model.predict_proba(user_encoded)[0][1]
        
        st.markdown("### Prediction Result")
        if prediction == 1:
            st.success(f"**Likely** to be a High-Value Purchase (Confidence: {probability:.1%})")
            st.balloons()
        else:
            st.warning(f"**Unlikely** to be a High-Value Purchase (Probability of High-Value: {probability:.1%})")

