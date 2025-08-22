import streamlit as st
import streamlit.components.v1 as stc
import pickle
import numpy as np
import plotly.graph_objects as go

# === Load Model ===
with open("GradientBoosting.pkl", "rb") as file:ß
    model = pickle.load(file)

# === Header HTML ===ß
html_temp = """<div style="background-color:#1E3D58;padding:15px;border-radius:10px">
                <h1 style="color:#F5F5F5;text-align:center">🚢 Titanic Survival Prediction</h1> 
                <h4 style="color:#F5F5F5;text-align:center">Data Science Project</h4> 
              </div>"""

desc_temp = """ 
### Welcome to Titanic Survival Prediction App 🎯  
This app uses a **Machine Learning Model** to predict whether a passenger would survive the Titanic disaster.  

📊 Dataset: [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)  
"""

# === Main App ===
def main():
    stc.html(html_temp)
    menu = ["Home", "Prediction App"]
    choice = st.sidebar.selectbox("📌 Menu", menu)

    if choice == "Home":
        st.subheader("🏠 Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)

    elif choice == "Prediction App":
        run_ml_app()

# === Prediction App ===
def run_ml_app():
    st.subheader("🔮 Passenger Information")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        sex = st.radio("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 25)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 30.0)
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    # Feature Encoding
    sex_encoded = 1 if sex == "female" else 0
    embarked_C = 1 if embarked == "C" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    # Feature Array
    features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_C, embarked_Q, embarked_S]])

    if st.button("🚀 Predict Survival"):
        prediction = model.predict(features)
        prob = model.predict_proba(features)[0][1]

        # Gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Survival Probability (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green" if prediction[0] == 1 else "red"}}
        ))

        if prediction[0] == 1:
            st.success(f"✅ Passenger Survived (Probability: {prob:.2%})")
        else:
            st.error(f"❌ Passenger Did Not Survive (Probability: {prob:.2%})")

        st.plotly_chart(fig, use_container_width=True)

# === Run App ===
if __name__ == "__main__":
    main()