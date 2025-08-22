import streamlit as st
import pickle
import numpy as np

# Load model
with open("GradientBoosting.pkl", "rb") as file:
    model = pickle.load(file)

# Judul Aplikasi
st.markdown(
    """
    <div style="background-color:#000;padding:10px;border-radius:10px">
        <h1 style="color:#fff;text-align:center">🚢 Titanic Survival Prediction</h1>
        <h4 style="color:#fff;text-align:center">Made with Gradient Boosting</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Menu
menu = ["Home", "Prediction App"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("🏠 Home")
    st.markdown(
        """
        ### Titanic Survival Prediction App  
        Aplikasi ini memprediksi kemungkinan seorang penumpang **selamat atau tidak**  
        berdasarkan informasi yang tersedia di dataset Titanic. Ⓒ By Edvan Tazul. 

        #### Data Source  
        Kaggle: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
        """
    )

elif choice == "Prediction App":
    st.subheader("🔮 Coba Prediksi")

    # Input user
    pclass = st.selectbox("Pclass (1=First, 2=Second, 3=Third)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    # Encode categorical
    sex = 1 if sex == "male" else 0
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    embarked = embarked_map[embarked]

    # Buat array input
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    if st.button("Predict"):
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]  # Probabilitas survival

        if prediction == 1:
            st.success(f"✅ Penumpang diprediksi SELAMAT dengan probabilitas {prob:.2f}")
        else:
            st.error(f"❌ Penumpang diprediksi TIDAK SELAMAT dengan probabilitas {prob:.2f}")

        # Progress bar survival probability
        st.write("📊 Survival Probability")
        st.progress(float(prob))

