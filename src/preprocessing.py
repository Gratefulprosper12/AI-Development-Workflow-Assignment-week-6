# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess(data):
    # Handle missing values
    data = data.fillna(method='ffill')

    # Example: Encode categorical features
    data = pd.get_dummies(data, drop_first=True)

    # Normalize numeric features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include='number'))

    return pd.DataFrame(data_scaled)
