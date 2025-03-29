import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data():
    df = pd.read_csv("Bank Customer Churn Prediction.csv")
    return df

# Feature Engineering
def preprocess_data(df):
    df = df.copy()
    df.dropna(inplace=True)
    
    # Customer Segmentation
    df['customer_segment'] = np.where(df['balance'] > df['balance'].median(), 'High Value', 'Regular')
    
    # Encode categorical variables
    label_encoders = {}
    original_mappings = {}
    categorical_columns = ['country', 'gender', 'customer_segment']
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            original_mappings[col] = dict(enumerate(le.classes_))
    
    # Creating an Engagement Score
    df['engagement_score'] = df['active_member'] * 2 + df['products_number'] + (df['tenure'] / df['tenure'].max())
    
    return df, label_encoders, original_mappings

def train_model(df):
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, accuracy, report, y_test, y_pred

def retention_strategy_simulator(churn_prob, strategy):
    impact = {
        'Discount': 0.15,
        'Loyalty Points': 0.10,
        'Personalized Offer': 0.20
    }
    return max(0, churn_prob - impact.get(strategy, 0))

def main():
    st.title("📊 Bank Customer Churn Prediction & Retention Strategy Simulator")
    df = load_data()
    df, label_encoders, original_mappings = preprocess_data(df)
    model, accuracy, report, y_test, y_pred = train_model(df)
    
    # Convert encoded values back to original labels for display
    for col in original_mappings:
        df[col] = df[col].map(original_mappings[col])
    
    st.sidebar.header("Filters")
    country_filter = st.sidebar.multiselect("Select Country", df['country'].unique(), default=df['country'].unique())
    gender_filter = st.sidebar.radio("Gender", ['All'] + list(df['gender'].unique()), index=0)
    
    filtered_df = df[df['country'].isin(country_filter)]
    if gender_filter != 'All':
        filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
    
    st.subheader("Dataset Overview")
    st.dataframe(filtered_df.head(10))
    
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df['churn'], palette='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    
    # Display Classification Report as a Table
    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df)
    
    # Precision-Recall Bar Chart
    st.subheader("Precision & Recall Comparison")
    metric_df = report_df[['precision', 'recall']].drop(index=['accuracy', 'macro avg', 'weighted avg'])
    metric_df.plot(kind='bar', figsize=(6, 4))
    st.pyplot(plt)
    
    st.subheader("Retention Strategy Simulator")
    churn_prob = st.slider("Select Predicted Churn Probability", 0.0, 1.0, 0.5)
    strategy = st.selectbox("Choose a Retention Strategy", ["None", "Discount", "Loyalty Points", "Personalized Offer"])
    
    new_churn_prob = retention_strategy_simulator(churn_prob, strategy)
    st.write(f"Predicted Churn Probability After Applying Strategy: {new_churn_prob:.2f}")
    
if __name__ == "__main__":
    main()