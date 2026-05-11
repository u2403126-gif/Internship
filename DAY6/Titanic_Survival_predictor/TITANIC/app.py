import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["Male", "Female"])

age = st.number_input("Age", min_value=0)

fare = st.number_input("Fare", min_value=0.0)

sibsp = st.number_input("Siblings/Spouse", min_value=0)

parch = st.number_input("Parents/Children", min_value=0)

embarked = st.selectbox(
    "Embarked",
    ["S", "C", "Q"]
)

sex = 0 if sex == "Male" else 1

embarked_map = {
    "S": 0,
    "C": 1,
    "Q": 2
}

embarked = embarked_map[embarked]

input_data = pd.DataFrame(
    [[
        pclass,
        sex,
        age,
        fare,
        sibsp,
        parch,
        embarked
    ]],
    columns=[
        'Pclass',
        'Sex',
        'Age',
        'Fare',
        'SibSp',
        'Parch',
        'Embarked'
    ]
)

input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):

    prediction = model.predict(input_data_scaled)[0]

    if prediction == 1:
        st.success("Survived")
    else:
        st.error("Not Survived")