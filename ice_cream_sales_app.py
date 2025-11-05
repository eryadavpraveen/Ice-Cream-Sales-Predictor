import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv


# Load environment variables (only needed for local runs)
load_dotenv()

# Securely load URI from environment
uri = os.getenv("MONGODB_URI")


# Connect to MongoDB
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Ice_Cream']
collection = db['Ice_Cream_Pred']

# def load_model():
#     with open("Ice_Cream_sales_final_model.pkl", "rb") as file:
#         model = pickle.load(file)
#     return model

def load_model():
    model_path = "Ice_Cream_sales_final_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()  # Stop app gracefully
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def predict_data(data):
    model = load_model()
    prediction = model.predict(data)
    return prediction[0]


def main():
    st.title("🍦 Ice Cream Sales Prediction App")
    st.write("Enter the temperature to predict expected ice cream sales.")

    # Input field
    temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)

    if st.button("Predict Sales"):
        # Convert input into a 2D structure (DataFrame)
        user_data = pd.DataFrame([[temp]], columns=["Temperature (°C)"])

        # Predict
        prediction = predict_data(user_data)

        user_data_copy=user_data.copy()
        user_data_copy['prediction']=round(float(prediction),2)
        collection.insert_one(user_data_copy.to_dict('records')[0])
        # Display result
        st.success(f"Predicted Ice Cream Sales: {prediction:.2f}")


if __name__ == "__main__":
    main()
