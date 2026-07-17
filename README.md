# Credit Card Fraud Detection

## Project Overview

This project predicts whether a credit card transaction is fraudulent using a Machine Learning model. The Streamlit application allows users to enter a credit card number, automatically retrieves the matching transaction details, and predicts whether the transaction is fraudulent or legitimate.

## Features

- Credit card number lookup
- Automatic transaction detail retrieval
- Fraud prediction using a trained ML model
- Streamlit web application

## Files

- app.py – Streamlit application
- app.ipynb – Data preprocessing, visualization, SMOTE, and model training
- fraud_detection_model.jb – Trained machine learning model
- label_encoder.jb – Saved label encoders
- dataset_small.csv – Sample dataset used by the deployed Streamlit application

## Dataset

The original dataset used for training was downloaded from the Kaggle Credit Card Fraud Detection dataset.

**Dataset Link:**  
[Kaggle Credit Card Fraud Detection Dataset]
(https://www.kaggle.com/mlg-ulb/creditcardfraud)

The original dataset (**dataset.csv**) is not included in this repository because it exceeds GitHub's file size limit.

The deployed Streamlit application uses **dataset_small.csv** only for transaction lookup and auto-fill.

To reproduce the complete training process in **app.ipynb**, download the original dataset from Kaggle and place it in the project folder as **dataset.csv**.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- LightGBM
- Geopy

