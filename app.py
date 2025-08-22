import streamlit as st
import pickle
import numpy as np
import inspect

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

# ============== MODEL LOADER (cached) ==============
@st.cache_resource
def load_model(path="GradientBoosting.pkl"):
    with open(path, "rb") as f:
        mdl = pickle.load(f)
    return mdl

model = load_model()

st.markdown(
    """
    <div style="background:#000;padding:12px;border-radius:12px">
      <h1 style="color:#fff;text-align:center">🚢 Titanic Survival Prediction</h1>
      <p style="color:#ccc;text-align:center;margin:0">Gradient Boosting (inference only)</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("Isi data penumpang di bawah ini lalu klik **Predict**.")

# ============== INPUT ==============
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25, step=1)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])
with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=0.5)

# One-hot boolean (0/1) — sesuaikan urutan dengan saat training!
sex_female = 1 if sex == "female" else 0
sex_male   = 1 if sex == "male" else 0
embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Urutan fitur (samakan dengan training!)
feature_vector = np.array([[
    pclass,
    sex_female, sex_male,
    embarked_c, embarked_q, embarked_s,
    age, sibsp, parch, fare
]], dtype=float)

# ============== PREDICT HELPERS ==============
def safe_predict(mdl, X):
    """
    Coba beberapa cara memanggil predict agar tahan banting.
    """
    # 1) panggil biasa
    try:
        return mdl.predict(X)
    except TypeError:
        pass
    # 2) pakai keyword (beberapa wrapper minta X=)
    try:
        return mdl.predict(X=X)
    except Exception as e:
        raise e

def safe_predict_proba(mdl, X):
    if hasattr(mdl, "predict_proba"):
        try:
            return mdl.predict_proba(X)
        except TypeError:
            return mdl.predict_proba(X=X)
    return None

# ============== ACTION ==============
if st.button("Predict"):
    try:
        y_pred = safe_predict(model, feature_vector)
        y = int(y_pred[0])

        proba = safe_predict_proba(model, feature_vector)
        if proba is not None:
            p_survive = float(proba[0][1])
        else:
            # fallback jika tidak ada predict_proba
            if hasattr(model, "decision_function"):
                score = float(model.decision_function(feature_vector))
                # skala kasar ke 0-1 (bukan probabilitas murni, hanya indikatif)
                p_survive = 1 / (1 + np.exp(-score))
            else:
                p_survive = 0.5  # konservatif

        if y == 1:
            st.success(f"✅ Diprediksi **SELAMAT** — Probabilitas: {p_survive*100:.2f}%")
        else:
            st.error(f"❌ Diprediksi **TIDAK SELAMAT** — Probabilitas: {p_survive*100:.2f}%")

        st.write("📊 Survival Probability")
        st.progress(min(max(p_survive, 0.0), 1.0))

    except Exception as e:
        st.error(f"Terjadi error saat inferensi: {e}")

# ============== DEBUG PANEL ==============
with st.expander("🔧 Debug (untuk diagnosis bila error)"):
    st.write("**type(model):**", type(model))
    st.write("**has predict:**", hasattr(model, "predict"))
    st.write("**has predict_proba:**", hasattr(model, "predict_proba"))
    try:
        st.write("**signature predict:**", str(inspect.signature(model.predict)))
    except Exception:
        st.write("signature predict: (tidak bisa dibaca)")
    try:
        if hasattr(model, "predict_proba"):
            st.write("**signature predict_proba:**", str(inspect.signature(model.predict_proba)))
    except Exception:
        st.write("signature predict_proba: (tidak bisa dibaca)")
    st.write("**Feature vector shape:**", feature_vector.shape)
    st.write("**Feature vector example:**", feature_vector.tolist())
