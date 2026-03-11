import streamlit as st
import pickle

# Load the saved model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("Model exported successfully")

