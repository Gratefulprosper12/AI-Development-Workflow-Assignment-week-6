# src/deploy_api.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return jsonify({'readmission_risk': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
