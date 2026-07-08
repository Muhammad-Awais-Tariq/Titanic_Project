import pandas as pd
import streamlit as st
import joblib
from transformer import TitanicTransformer
import numpy as np

model = joblib.load("tree_pipeline.joblib")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("Titanic Survival Predictor")
st.caption("Fill in passenger details and see if they would have survived.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

with col2:
    sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, value=0)
    parch = st.number_input("Parents/Children aboard", min_value=0, value=0)
    fare = st.number_input("Fare paid", min_value=0.0, value=32.0)
    cabin = st.text_input("Cabin (leave blank if unknown)", "")

name = st.text_input("Name (include 'Mrs' if applicable)", "Mr. John Smith")

st.divider()

predict_clicked = st.button("Predict Survival", use_container_width=True)

if predict_clicked:
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

    st.divider()

    if prediction == 1:
        st.success("The passenger survived!")
        st.image("https://media.giphy.com/media/g9582DNuQppxC/giphy.gif")
    else:
        st.error("The passenger did not survive.")
        st.image("https://media.giphy.com/media/10KJIBMXN8Vt4Y/giphy.gif")