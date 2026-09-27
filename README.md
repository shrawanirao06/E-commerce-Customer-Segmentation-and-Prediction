# E-commerce Customer Segmentation and Prediction

## Project Overview

This project focuses on analyzing e-commerce customer purchasing behavior to identify different customer segments and predict whether customers are likely to make a future purchase.

The project uses **RFM (Recency, Frequency, Monetary) analysis**, **K-Means clustering**, and **Gradient Boosting Machine Learning** to support customer targeting, retention, and marketing strategies.

## Objectives

- Analyze customer purchasing behavior.
- Perform data cleaning and preprocessing.
- Create customer-level RFM features.
- Segment customers based on purchasing patterns.
- Predict future customer purchase behavior.
- Identify high-value, active, moderate, and at-risk customers.
- Provide a simple interactive Streamlit application.

## Dataset

The project uses an e-commerce transaction dataset containing information such as:

- Invoice Number
- Stock Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

The dataset contains transaction-level information that is transformed into customer-level features for analysis and prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means Clustering
- Gradient Boosting
- Joblib
- Streamlit
- Jupyter Notebook

## Project Workflow

```text
Raw E-commerce Data
        ↓
Data Cleaning
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
RFM Analysis
        ↓
Customer Segmentation
        ↓
K-Means Clustering
        ↓
Future Purchase Prediction
        ↓
Gradient Boosting Model
        ↓
Streamlit Application
