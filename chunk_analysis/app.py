import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

with open("churn_pipeline.pkl","rb") as f:
    pipeline = pickle.load(f)

data = pd.read_csv("cleaned_churn.csv")

st.set_page_config(page_title="Telecom Churn Prediction web app", layout="wide")

page = st.sidebar.radio("Select Page", ["Top 10 Features", "Predict the Churn"])

model = pipeline.named_steps['model']
if hasattr(model, 'feature_importances_'):
    try:
        feature_names = pipeline[:-1].get_feature_names_out()
    except:
        feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
    feat_imp = pd.Series(model.feature_importances_, index=feature_names)
    top_features = feat_imp.sort_values(ascending=False).head(10)


if page == "Predict the Churn":

    st.sidebar.title("📊 Customer Churn Prediction")
    st.markdown("Ready to predict if they leave or stay 🍀. Fill the customer detail")

    # model = pipeline.named_steps

    state_list = sorted(data['State'].unique())
    area_list = sorted(data['Area code'].unique())
    cust_service_list = sorted(data['Customer service calls'].unique())


    #Categorical Cols
    state = st.sidebar.selectbox("State", state_list)
    area_code = st.sidebar.selectbox("Area Code", area_list)

    #Binary Cols
    intl_plan = st.sidebar.selectbox("Intl. Plan", ["Yes", "No"])
    voice_plan = st.sidebar.selectbox("Voice mail plan", ["Yes", "No"])

    #Numerical Cols
    account_length = st.sidebar.number_input("Account length", min_value=0, value=100)
    day_calls = st.sidebar.number_input("Total day calls", min_value=0, value=100)
    day_charge = st.sidebar.number_input("Total day charge", min_value=0.0, value=30.0)
    eve_calls = st.sidebar.number_input("Total eve calls", min_value=0, value=100)
    eve_charge = st.sidebar.number_input("Total eve charge", min_value=0.0, value=20.0)
    night_calls = st.sidebar.number_input("Total night calls", min_value=0, value=100)
    night_charge = st.sidebar.number_input("Total night charge", min_value=0.0, value=15.0)
    intl_calls = st.sidebar.number_input("Total intl calls", min_value=0, value=5)
    intl_charge = st.sidebar.number_input("Total intl charge", min_value=0.0, value=2.7)
    cust_service_calls = st.sidebar.selectbox("Customer service calls",cust_service_list, index=cust_service_list.index(0))

    intl_plan = 1 if intl_plan == "Yes" else 0
    voice_plan = 1 if voice_plan == "Yes" else 0

    user_input = pd.DataFrame({
        "State": [state],
        "Area code" : [area_code],
        "International plan": [intl_plan],
        "Voice mail plan": [voice_plan],
        "Account length": [account_length],
        "Total day calls": [day_calls],
        "Total day charge": [day_charge],
        "Total eve calls": [eve_calls],
        "Total eve charge": [eve_charge],
        "Total night calls": [night_calls],
        "Total night charge": [night_charge],
        "Total intl calls": [intl_calls],
        "Total intl charge": [intl_charge],
        "Customer service calls": [cust_service_calls]
    })


    if st.button("Predict"):
        proba = pipeline.predict_proba(user_input)[0] 
        st.write(f"Probability of Not Churn: {proba[0]:.2f}")
        st.write(f"Probability of Churn: {proba[1]:.2f}")

        threshold = 0.3
        prediction = 1 if proba[1] > threshold else 0
        st.success("Churn" if prediction == 1 else "Not Churn")

        # Risk level
        if proba[1] < 0.3:
            risk = "Low Risk"
        elif proba[1] < 0.6:
            risk = "Medium Risk"
        else:
            risk = "High Risk"
        st.info(f"Churn Risk Level: {risk}")

    model = pipeline.named_steps['model']

elif page == "Top 10 Features":
    st.title("📊 Top 10 Features Affecting Churn")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)
