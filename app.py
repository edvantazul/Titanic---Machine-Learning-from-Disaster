import streamlit as st
import pickle
import numpy as np

# --- Load model yang sudah ditraining di notebook ---
with open("GradientBoosting.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction App")

# --- Input User ---
pclass = st.selectbox("Pclass (Passenger Class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# --- One-hot encoding manual (pastikan sesuai training) ---
sex_female = 1 if sex == "female" else 0
sex_male = 1 if sex == "male" else 0

embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# --- Urutan harus sama dengan waktu training ---
features = np.array([[pclass, sex_female, sex_male,
                      embarked_c, embarked_q, embarked_s,
                      age, sibsp, parch, fare]], dtype=float)

# --- Prediksi ---
if st.button("Predict"):
    try:
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        if prediction == 1:
            st.success(f"✅ Penumpang diprediksi **SELAMAT** dengan probabilitas {proba[1]*100:.2f}%")
        else:
            st.error(f"❌ Penumpang diprediksi **TIDAK SELAMAT** dengan probabilitas {proba[0]*100:.2f}%")
    except Exception as e:
        st.error(f"Terjadi error: {e}")
