
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

# Load saved dictionary
@st.cache_resource
def load_package():
    return joblib.load("titanic_model.pkl")

package = load_package()

# Extract trained model and encoders
model = package["model"]
encoders = package["encoders"]
features = package["features"]

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Input fields must match the trained model's categories
pclass = st.selectbox(
    "Passenger Class",
    ["First", "Second", "Third"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

sibling = st.selectbox(
    "Number of Siblings",
    ["Zero", "One", "Two", "Three"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["Southampton", "Queenstown", "Cherbourg"]
)

if st.button("Predict Survival"):

    input_values = {
        "PClass": pclass,
        "Gender": gender,
        "Sibling": sibling,
        "Embarked": embarked
    }

    # Encode inputs using the saved encoders
    encoded_values = {}

    for column, value in input_values.items():
        encoded_values[column] = encoders[column].transform([value])[0]

    # Arrange features in the same order used during training
    input_df = pd.DataFrame(
        [[encoded_values[column] for column in features]],
        columns=features
    )

    # Predict using the actual trained model
    prediction = model.predict(input_df)[0]

    # Convert prediction back to its original label
    result = encoders["Survived"].inverse_transform(
        [int(prediction)]
    )[0]

    if result == "Yes":
        st.success("Prediction: Passenger Survived! 🎉")
    else:
        st.error("Prediction: Passenger Did Not Survive.")
