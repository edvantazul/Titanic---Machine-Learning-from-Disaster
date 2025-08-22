import streamlit as st
import streamlit.components.v1 as stc
import pickle
import numpy as np
import plotly.graph_objects as go

# === Load Gradient Boosting Model ===
with open("GradientBoosting.pkl", "rb") as file:
    gb_model = pickle.load(file)

# === HTML Template ===
html_temp = """
<div style="background-color:#003366;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Titanic Survival Prediction App</h1> 
    <h4 style="color:#fff;text-align:center">Powered by Gradient Boosting</h4> 
</div>
"""

desc_temp = """
### 🚢 Titanic Prediction App  
This app predicts whether a passenger would **survive or not** based on input features.

#### ⚙️ Model Used
Gradient Boosting Classifier trained on Titanic dataset.
"""

# === Prediction Function ===
def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = gb_model.predict(features)[0]
    prob = gb_model.predict_proba(features)[0][1]
    return prediction, prob

# === Main App ===
def main():
    stc.html(html_temp)
    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)

    elif choice == "Prediction":
        st.subheader("Passenger Information")

        # Input fields
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 25)
        sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 32.0)
        embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

        # Preprocess categorical to numeric (sesuaikan dengan training)
        sex_num = 1 if sex == "female" else 0
        embarked_map = {"C": 0, "Q": 1, "S": 2}
        embarked_num = embarked_map[embarked]

        # Features (urutan harus sama seperti training)
        features = [pclass, sex_num, age, sibsp, parch, fare, embarked_num]

        if st.button("Predict Survival"):
            prediction, prob = predict(features)

            if prediction == 1:
                st.success(f"✅ Passenger Survives with probability {prob:.2%}")
            else:
                st.error(f"❌ Passenger Does Not Survive (probability {prob:.2%})")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Survival Probability (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
