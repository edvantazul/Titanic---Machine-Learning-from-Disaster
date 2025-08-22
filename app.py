import streamlit as st
import pickle
import numpy as np

# Load model
with open("GradientBoosting.pkl", "rb") as file:
    model = pickle.load(file)

# --- Aplikasi ---
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.markdown(
    """
    <div style="background-color:#000;padding:10px;border-radius:10px">
        <h1 style="color:#fff;text-align:center">🚢 Titanic Survival Prediction App</h1>
        <h4 style="color:#fff;text-align:center">Gradient Boosting Model</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Isi data penumpang di bawah ini untuk memprediksi apakah mereka **selamat atau tidak**.")

# --- Input User ---
pclass = st.selectbox("Pclass (Passenger Class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)

# Encode sex sesuai training
sex = 1 if sex == "male" else 0

# Gabungkan feature ke array 2D
features = np.array([[pclass, sex, age, sibsp, parch, fare]])

# --- Prediksi ---
if st.button("Predict"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    if prediction == 1:
        st.success(f"✅ Penumpang diprediksi **SELAMAT** dengan probabilitas {proba[1]*100:.2f}%")
    else:
        st.error(f"❌ Penumpang diprediksi **TIDAK SELAMAT** dengan probabilitas {proba[0]*100:.2f}%")

