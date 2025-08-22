import streamlit as st
import pandas as pd
import pickle

# --- Load model ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Prediksi Keselamatan Penumpang Titanic (Gradient Boosting) by Edvan")

# --- Input user ---
pclass = st.selectbox("Pclass", [1,2,3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["C","Q","S"])

# --- Preprocessing input ---
sex = 0 if sex == "male" else 1
embarked = {"C":0, "Q":1, "S":2}[embarked]

input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# --- Prediksi ---
if st.button("Prediksi"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ Penumpang kemungkinan SELAMAT ({prob:.2f}% confidence)")
    else:
        st.error(f"❌ Penumpang kemungkinan TIDAK selamat ({100-prob:.2f}% confidence)")
