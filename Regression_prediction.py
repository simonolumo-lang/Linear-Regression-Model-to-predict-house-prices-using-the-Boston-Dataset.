import streamlit as st
import numpy as np
import pickle

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# If you used scaling, uncomment these lines:
# with open("scaler.pkl", "rb") as file:
#     scaler = pickle.load(file)

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction System",    layout="centered"
)

# -------------------------------------------------
# Project Title
# -------------------------------------------------
st.title("Boston Housing Price Prediction System")

# -------------------------------------------------
# Model Description
# -------------------------------------------------
st.header("Model Description")
st.write("""
This web application predicts house prices using a *Linear Regression model*
trained on the *Boston Housing dataset*.

- *Dataset used:* Boston Housing Dataset
- *Target variable:* PRICE
- *Machine learning algorithm used:* Linear Regression
- *Purpose of the prediction system:* To estimate house prices based on housing features
""")

# -------------------------------------------------
# User Input Section
# -------------------------------------------------
st.header("Enter House Feature Values")

CRIM = st.number_input("CRIM - Per capita crime rate by town", min_value=0.0, value=0.10, format ="%.5f")
ZN = st.number_input("ZN - Proportion of residential land zoned for large lots", min_value=0.0, value=0.0, format ="%.2f")
INDUS = st.number_input("INDUS - Proportion of non-retail business acres per town", min_value=0.0, value=8.0, format ="%.3f")
CHAS = st.selectbox("CHAS - Charles River dummy variable", [0, 1])
NOX = st.number_input("NOX - Nitric oxide concentration", min_value=0.0, value=0.50, format ="%.5f")
RM = st.number_input("RM - Average number of rooms per dwelling", min_value=0.0, value=6.0, format ="%.4f")
AGE = st.number_input("AGE - Proportion of owner-occupied units built before 1940", min_value=0.0, value=65.0, format ="%.2f")
DIS = st.number_input("DIS - Weighted distances to employment centres", min_value=0.0, value=4.0, format ="%.4f")
RAD = st.number_input("RAD - Index of accessibility to radial highways", min_value=0.0, value=4.0)
TAX = st.number_input("TAX - Full-value property tax rate per $10,000", min_value=0.0, value=300.0)
PTRATIO = st.number_input("PTRATIO - Pupil-teacher ratio by town", min_value=0.0, value=18.0)
B = st.number_input("B - Calculated dataset variable", min_value=0.0, value=390.0)
LSTAT = st.number_input("LSTAT - Percentage lower status of the population", min_value=0.0, value=12.0)

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.header("Prediction")

if st.button("Predict"):
    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    # If scaler was used during training:
    # input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: {prediction[0]:.2f}")

# -------------------------------------------------
# Model Use Cases
# -------------------------------------------------
st.header("Model Use Cases")
st.write("""
Real-World Applications of the Model
1. Real Estate Price Estimation

This model can help estimate the price of houses based on their characteristics.
In the Boston Housing dataset, factors such as number of rooms (RM), crime rate (CRIM), property tax (TAX), and population characteristics (LSTAT) influence house prices.

Using the trained Linear Regression model, a user can enter these values into the Streamlit application and receive a predicted house price instantly.

This can help:

real estate agents estimate property value

homeowners understand the potential price of their houses

buyers compare property prices before making decisions

2. Property Market Analysis

The model can also be used to analyze how different factors affect housing prices in a city or region.

For example, the model shows that:

houses with more rooms (RM) tend to have higher prices

areas with higher crime rates (CRIM) or higher lower-status population percentages (LSTAT) may have lower housing prices

By analyzing these relationships, property developers, economists, and researchers can better understand housing market trends and factors that influence property values.

3. Educational Demonstration of Machine Learning

This project is a good example of how machine learning works from start to finish.

It demonstrates the complete workflow:

loading and understanding data

performing exploratory data analysis

preparing the dataset

training a Linear Regression model

evaluating the model

deploying it using a Streamlit web application

Students and beginners can use this project to learn how machine learning models are developed and deployed into real applications.

4. Decision Support for Understanding Price-Influencing Factors

The model can help people understand which factors influence housing prices the most.

For example, the analysis in this project showed that:

RM (number of rooms) has a strong positive effect on house prices

LSTAT (percentage of lower-status population) has a strong negative effect on house prices


In conclusion, this model demonstrates how machine learning can be used to analyze housing data, predict property prices, and support decision-making in the real estate industry
""")