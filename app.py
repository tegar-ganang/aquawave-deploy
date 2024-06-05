from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
ph_model = joblib.load('ph_model.pkl')
tds_model = joblib.load('tds_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Generate daily inputs
    hours = list(range(6, 22))
    day = [data['day']] * len(hours)
    month = [data['month']] * len(hours)
    year = [data['year']] * len(hours)

    features = np.array([hours, day, month, year]).T
    
    ph_predictions = ph_model.predict(features)
    tds_predictions = tds_model.predict(features)

    results = [
        {'hour': hour, 'ph': ph, 'tds': tds} 
        for hour, ph, tds in zip(hours, ph_predictions, tds_predictions)
    ]

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
