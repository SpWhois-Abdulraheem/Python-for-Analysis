# -*- coding: utf-8 -*-
"""


@author: DELL
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset URL
dataset_url = "https://data.cdc.gov/api/views/mssc-ksj7/rows.xml?accessType=DOWNLOAD"

# Data Ingestion
def load_data_from_xml(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the XML
            root = ET.fromstring(response.content)
            data = []
            # Extract rows
            for row in root.findall(".//row"):
                record = {child.tag: child.text for child in row}
                data.append(record)
            # Convert to DataFrame
            df = pd.DataFrame(data)
            print("Data loaded successfully!")
            return df
        else:
            print(f"Error fetching data: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

#Data Preprocessing
def preprocess_data(data):
    # Converting `totalpopulation` to numeric
    if 'totalpopulation' in data.columns:
        data['totalpopulation'] = pd.to_numeric(data['totalpopulation'], errors='coerce')
        print("\nSummary Statistics for 'totalpopulation':")
        print(data['totalpopulation'].describe())

    # Converting range values in `access2_crudeprev` to numeric (extract lower bound)
    if 'access2_crudeprev' in data.columns:
        data['access2_crudeprev'] = data['access2_crudeprev'].str.extract(r'(\d+\.\d+)').astype(float)
        print("\nSummary Statistics for 'access2_crudeprev':")
        print(data['access2_crudeprev'].describe())

    # Dropping rows with missing `stateabbr` or `totalpopulation`
    data = data.dropna(subset=['stateabbr', 'totalpopulation'])
    print("\nMissing Values After Dropping:")
    print(data.isnull().sum())
    return data

# Visualization of the datas
def visualize_data(data):
    if 'totalpopulation' in data.columns:
        data['totalpopulation'].hist()
        plt.title('Total Population Distribution')
        plt.xlabel('Population')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("'totalpopulation' column not found for visualization.")

# Predictive Model
def predictive_model(data):
    # Ensureing `stateabbr` and `totalpopulation` have no missing values
    if 'stateabbr' in data.columns and 'totalpopulation' in data.columns:
        # Encode `stateabbr` as numeric
        data['stateabbr'] = data['stateabbr'].astype('category').cat.codes

        # Defining features and target
        features = data[['stateabbr']]
        target = data['totalpopulation']

        # Debugging: Print features and target
        print("\nFeatures (stateabbr):")
        print(features.head())
        print("\nTarget (totalpopulation):")
        print(target.head())

        # Checking if there are valid samples
        if len(features) == 0 or len(target) == 0:
            print("No valid data available for modeling.")
            return

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        # Checking for valid split
        if len(X_train) == 0 or len(y_train) == 0:
            print("Train set is empty after splitting. Check your data.")
            return

        # Fit a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Making predictions and evaluation
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"\nModel Mean Squared Error: {mse}")
    else:
        print("Required columns for predictive modeling are not available.")

# Pipeline Execution
data = load_data_from_xml(dataset_url)

if data is not None:
    print("\nColumns in the Dataset:")
    print(data.columns)

    data = preprocess_data(data)
    visualize_data(data)
    predictive_model(data)
else:
    print("Failed to load data.")
