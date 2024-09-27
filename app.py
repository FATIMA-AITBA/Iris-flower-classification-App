import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_artifacts():
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

st.title("🌸 Iris Flower Classification App")
st.write("""
This app predicts the **Iris flower** species based on its measurements.
""")

st.sidebar.header("Input Flower Features")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.0)
    data = {'SepalLengthCm': sepal_length,
            'SepalWidthCm': sepal_width,
            'PetalLengthCm': petal_length,
            'PetalWidthCm': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

def predict_species(input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    return prediction, prediction_proba

prediction, prediction_proba = predict_species(input_df)

species = label_encoder.inverse_transform(prediction)[0]

st.subheader('Prediction')
st.write(f"The predicted species is: **{species}**")

st.subheader('Prediction Probability')
prob_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
st.write(prob_df)

st.subheader('Visualizations')

if st.checkbox('Show Pairplot'):
    try:
        image = Image.open('pairplot.png')
        st.image(image, caption='Pairplot of Iris Dataset', use_column_width=True)
    except FileNotFoundError:
        st.write("Pairplot image not found.")

if st.checkbox('Show Confusion Matrix'):
    try:
        image = Image.open('confusion_matrix.png')
        st.image(image, caption='Confusion Matrix', use_column_width=True)
    except FileNotFoundError:
        st.write("Confusion matrix image not found.")
