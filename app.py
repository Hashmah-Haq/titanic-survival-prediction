
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()

st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict Survival"):
    passenger = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }])

    try:
        prediction = model.predict(passenger)[0]

        if prediction == 1:
            st.success("Prediction: Passenger Survived!")
        else:
            st.error("Prediction: Passenger Did Not Survive!")

    except Exception as e:
        st.error("Input format does not match the trained model.")
        st.write(str(e))
