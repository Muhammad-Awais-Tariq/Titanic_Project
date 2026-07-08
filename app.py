import pandas as pd
import streamlit as st
import joblib
from transformer import TitanicTransformer 
import numpy as np

model = joblib.load("tree_pipeline.joblib")
st.title("Titanic Prediction")

pclass = st.selectbox("Pclass", [1, 2, 3])
name = st.text_input("Name (include 'Mrs' if applicable)", "Mr. John Smith")
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
cabin = st.text_input("Cabin (leave blank if unknown)", "")
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "PassengerId": 0,
        "Pclass": pclass,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "N/A",
        "Fare": fare,
        "Cabin": cabin if cabin.strip() != "" else np.nan,
        "Embarked": embarked
    }])

    prediction = model.predict(input_df)[0]
    st.write("Survived!" if prediction == 1 else "Did not survive.")