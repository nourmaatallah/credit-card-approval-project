import streamlit as st
import numpy as np
import pandas as pd
import joblib

model=joblib.load('credit_card_model.pkl')
model_columns=joblib.load('model_columns.pkl')

def predict_approvals(genre, age, debt, married, bank_customer, years_employed, prior_default, employed, credit_score,  zip_code, income, industry, citizen):
    
    input_data = {
        "Gender": 1 if genre == "Male" else 0,
        "Age": age,
        "Debt": debt,
        "Married": 1 if married == "Yes" else 0,
        "BankCustomer": 1 if bank_customer == "Yes" else 0,
        "YearsEmployed": years_employed,
        "PriorDefault": 1 if prior_default == "Yes" else 0,
        "Employed": 1 if employed == "Yes" else 0,
        "CreditScore": credit_score,
        "ZipCode": zip_code,
        "Income": income,
    }
    # Industry encoding
    industry_columns = [col for col in model_columns if col.startswith("Industry_")]
    for col in industry_columns:
        input_data[col] = 1 if col == f"Industry_{industry}" else 0

    # Citizen encoding
    citizen_columns = [col for col in model_columns if col.startswith("Citizen_")]
    for col in citizen_columns:
        input_data[col] = 1 if col == f"Citizen_{citizen}" else 0
    
    user_input = pd.DataFrame([input_data])[model_columns]
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0]
    return prediction,probability



st.title("Credit Cards Approval App")
html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Your Credit Card Approval Predictor </h2>
    </div>
    """

st.markdown(html_temp, unsafe_allow_html=True)

genre=st.selectbox("Gender", ["Male",'Female'])
age=st.number_input("Age", min_value=18, max_value=100)
debt=st.number_input("Debt", min_value=0.0)
married=st.selectbox("Married", ["Yes","No"])
bank_customer=st.selectbox("Bank Customer", ["Yes","No"])
years_employed=st.number_input("Years Employed", min_value=0.0)
prior_default=st.selectbox("Prior Default", ["Yes","No"])
employed=st.selectbox("Employed", ["Yes","No"])
credit_score=st.number_input("Credit Score", min_value=0)
zip_code=st.number_input("Zip Code", min_value=0)
income=st.number_input("Income", min_value=0.0)

industry=st.selectbox("Industry", ["Construction", "Education", "Energy & Utilities", "Finances", "Government & Public Administration", "Health Care",
                         "Information & Technology", "Leisure & Hospitality", "Manufacturing", "Professional & Buisiness Services", 
                         "Real Estate", "Retail Trade", "Transportation","Agriculture"])
    
citizen=st.selectbox("Citizen", ["ByBirth","ByOtherMeans", "Temporary"])
result=""
if st.button("Predict"):
    result,probability = predict_approvals(genre, age, debt, married, bank_customer, years_employed,
                               prior_default, employed, credit_score,
                               zip_code, income, industry, citizen)
    approval_proba=probability[1]*100
    rejection_proba=probability[0]*100
    st.success('The output is {}'.format(result))
    st.markdown("### 📊 Model Confidence")
    st.write(f"Approval Probability: {approval_proba:.2f}%")
    st.write(f"Rejection Probability: {rejection_proba:.2f}%")
    st.progress(int(approval_proba))







