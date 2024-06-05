from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
ph_model = joblib.load('random_forest_ph_model.pkl')
tds_model = joblib.load('random_forest_tds_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([[
        data['hour'],
        data['day'],
        data['month'],
        data['year']
    ]])
    
    ph_prediction = ph_model.predict(features)
    tds_prediction = tds_model.predict(features)

    return jsonify({
        'ph': ph_prediction[0],
        'tds': tds_prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
