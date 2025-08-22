import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# === Load model yang sudah ditraining ===
with open("gradient_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🚢 Titanic Survival Prediction", page_icon="🚢", layout="centered")

# === Header ===
st.title("🚢 Titanic Survival Prediction")
st.markdown("Masukkan data penumpang untuk memprediksi kemungkinan **selamat atau tidak**.")

# === Input Form ===
with st.form("titanic_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Pclass", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 25)
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)

    with col2:
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 32.0)
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    submitted = st.form_submit_button("Predict 🎯")

# === Preprocessing Input ===
if submitted:
    sex = 0 if sex == "male" else 1
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    embarked = embarked_map[embarked]

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    # === Predict ===
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1] * 100  # probabilitas selamat

    # === Hasil ===
    st.subheader("📊 Hasil Prediksi")
    if prediction == 1:
        st.success(f"✅ Penumpang ini **DIPREDIKSI SELAMAT** ({proba:.2f}% kemungkinan).")
    else:
        st.error(f"❌ Penumpang ini **TIDAK SELAMAT** ({proba:.2f}% kemungkinan selamat).")

    # Tambah chart probabilitas
    st.progress(int(proba))
    st.markdown(f"Probabilitas Selamat: **{proba:.2f}%**")

    st.bar_chart(pd.DataFrame({
        "Probability": [100-proba, proba]
    }, index=["Tidak Selamat", "Selamat"]))
