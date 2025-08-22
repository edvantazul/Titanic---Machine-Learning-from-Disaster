import streamlit as st
import pickle   # <<< tambahkan ini
import numpy as np

# --- Load trained model ---
with open("GradientBoosting.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")

# --- Input form ---
pclass   = st.selectbox("Passenger Class", [1, 2, 3])
sex      = st.selectbox("Sex", ["male", "female"])
age      = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp    = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch    = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare     = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# --- Encode input ---
sex_map = {"male": 0, "female": 1}
embarked_map = {"C": 0, "Q": 1, "S": 2}

features = np.array([[pclass, sex_map[sex], age, sibsp, parch, fare, embarked_map[embarked]]])

# --- Predict ---
if st.button("Predict"):
    try:
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]  # [prob_tidak_selamat, prob_selamat]
        survival_chance = proba[1] * 100

        if prediction == 1:
            st.success(f"✅ Penumpang diprediksi **SELAMAT** ({survival_chance:.2f}% kemungkinan)")
        else:
            st.error(f"❌ Penumpang diprediksi **TIDAK SELAMAT** ({survival_chance:.2f}% kemungkinan selamat)")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

